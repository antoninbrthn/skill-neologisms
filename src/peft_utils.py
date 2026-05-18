"""
Utilities for working with PEFT (Parameter-Efficient Fine-Tuning) models.

Supports various PEFT methods like LoRA, Prefix Tuning, Prompt Tuning, etc.
"""

import os
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    TaskType,
    PeftModel,
    PeftConfig,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.models.model_utils import get_mean_emb

PEFT_TYPE_TO_WARMUP = {
    "lora": False,
    "prefix_tuning": True,
    "prompt_tuning": True,
}


def get_peft_config(peft_cfg: DictConfig) -> Any:
    """
    Create a PEFT config based on the configuration.

    Args:
        peft_cfg: Configuration dict with 'type' and 'params' keys

    Returns:
        PEFT config object (LoraConfig, PromptTuningConfig, etc.)

    Example config:
        peft:
          type: "lora"
          params:
            r: 8
            lora_alpha: 32
            target_modules: ["c_attn", "c_proj"]
            lora_dropout: 0.1
    """
    peft_type = peft_cfg.type.lower()
    params = OmegaConf.to_container(peft_cfg.params, resolve=True) if "params" in peft_cfg else {}

    # Add task_type if not specified (default to CAUSAL_LM)
    if "task_type" not in params:
        params["task_type"] = TaskType.CAUSAL_LM

    if peft_type == "lora":
        return LoraConfig(**params)
    elif peft_type == "prefix_tuning":
        return PrefixTuningConfig(**params)
    elif peft_type == "prompt_tuning":
        return PromptTuningConfig(**params)
    elif peft_type == "p_tuning":
        return PromptEncoderConfig(**params)
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")


def create_peft_model(
    model: PreTrainedModel,
    peft_cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    skip_pt_init: bool = False,
    skip_warmup: bool = False,
) -> PeftModel:
    """
    Create a PEFT model from a base model and config.

    Args:
        model: Base pretrained model
        peft_cfg: PEFT configuration
        checkpoint_path: Optional path to load existing PEFT adapter weights
        skip_pt_init: Skip prompt tuning embedding init from pretrain ops
                      (useful for other experiments that don't have those tokens)
        skip_warmup: Skip PEFT warmup step (useful for large models where warmup is expensive)

    Returns:
        PEFT model with adapter attached
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading PEFT adapter from checkpoint: {checkpoint_path}")
        # Load the adapter config and model
        peft_model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded PEFT adapter: {peft_model.peft_config}")
    else:
        print("Creating new PEFT adapter from config")
        peft_config = get_peft_config(peft_cfg)
        print(f"PEFT config: {peft_config}")
        peft_model = get_peft_model(model, peft_config)

        # For prompt tuning: init embedding via PRETRAINED_OPS embeddings, similar to skill neologisms
        if peft_cfg.type.lower() == "prompt_tuning" and not skip_pt_init:
            init_pt_embeddings_from_ops(peft_model, tokenizer=tokenizer)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    # Warm-up if needed
    peft_type = peft_cfg.type.lower()

    if PEFT_TYPE_TO_WARMUP.get(peft_type, False) and not skip_warmup:
        print(f"Warming up PEFT model of type: {peft_type}")
        peft_warmup(peft_model, tokenizer=tokenizer, device="cuda")

    return peft_model


def init_pt_embeddings_from_ops(peft_model, tokenizer):
    from sequence_map_experiment.data import PRETRAIN_OPS

    prefix_embs = peft_model.prompt_encoder.default.embedding.weight
    print(f"Re-initialising prompt token embeddings...")
    print(f"  Current prefix embeddings: {prefix_embs}")
    emb = peft_model.get_input_embeddings().weight
    mean_emb = get_mean_emb(emb, tokenizer, word_list=PRETRAIN_OPS)
    with torch.no_grad():
        for tid in range(peft_model.prompt_encoder.default.embedding.num_embeddings):
            prefix_embs[tid] = mean_emb.clone()
    print(f"Re-initialised prompt token embeddings from mean of {len(PRETRAIN_OPS)} pretrain ops embeddings: {PRETRAIN_OPS}")
    print(f"  New prefix embeddings: {peft_model.prompt_encoder.default.embedding.weight.data}")


def save_peft_adapter(
    model: PeftModel,
    save_path: str,
) -> None:
    """
    Save PEFT adapter weights.

    Args:
        model: PEFT model
        save_path: Directory to save adapter
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Saved PEFT adapter to: {save_path}")


def load_peft_adapter(
    model: PreTrainedModel,
    adapter_path: str,
) -> PeftModel:
    """
    Load a PEFT adapter onto a base model.

    Args:
        model: Base pretrained model
        adapter_path: Path to saved adapter

    Returns:
        PEFT model with loaded adapter
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded PEFT adapter from: {adapter_path}")
    peft_model.print_trainable_parameters()

    return peft_model


def peft_warmup(
    peft_model,
    tokenizer,
    device="cuda",
):
    """
    Some PEFT models run into errors when used for generation without prior training.
    This function does a dummy training step that doesn't change any parameters.
    """
    peft_model.eval()
    peft_model.to(device)

    # 1. Snapshot all parameters (base + PEFT)
    before_params = {}
    for name, param in peft_model.named_parameters():
        before_params[name] = param.detach().cpu().clone()

    # 2. Dummy dataset (minimal but valid for SFTTrainer)
    dummy_dataset = Dataset.from_list([{"prompt": "DUMMY[OUTPUT]", "completion": "0"}])

    # 3. Zero-update SFT config
    training_args = SFTConfig(
        output_dir="./_peft_warmup",
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=0.0,  # guarantees no gradient update
        weight_decay=0.0,
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dummy_dataset,
    )

    # 4. Run warm-up
    trainer.train()

    # 5. Assert parameter equality
    for name, param in peft_model.named_parameters():
        before = before_params[name]
        after = param.detach().cpu()

        if not torch.equal(before, after):
            diff = (before - after).abs().max().item()
            raise AssertionError(f"Parameter changed during PEFT warm-up: {name} " f"(max |Δ| = {diff})")

    print("PEFT warm-up completed: all parameters unchanged.")


def load_peft_model_from_config(cfg, model_dir=None, peft_adapter_path=None):
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    # Load pretrained base model
    if model_dir is None:
        from sequence_map_experiment.config import OUTPUT_DIR

        pretrained_path = cfg.pretrained.checkpoint
        local_model_path = os.path.join(OUTPUT_DIR, pretrained_path)
        if os.path.exists(local_model_path):
            model_dir = local_model_path
        else:
            model_dir = pretrained_path
    print(f"Loading base model from {model_dir}...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, padding_side="left")

    # Create PEFT adapter
    print(f"Creating PEFT adapter...")
    peft_model = create_peft_model(
        model=model,
        peft_cfg=cfg.peft,
        checkpoint_path=peft_adapter_path,
        tokenizer=tokenizer,
    )
    print(f"PEFT model loaded from {peft_adapter_path}")

    return peft_model, tokenizer
