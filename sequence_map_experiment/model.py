import os
import yaml
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftConfig, PeftModel


def load_any_model(checkpoint_path, cfg=None):
    """
    Supports:
      - local full models
      - local LoRA adapters
      - HF full models
      - HF LoRA adapters
    """

    print(f"\nLoading model from: {checkpoint_path}")

    # For local model
    if cfg is None:
        config_path = os.path.join(checkpoint_path, "..", "..", "config.yaml")
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}...")
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            print(f"  Config keys: {list(cfg.keys())}")
        else:
            cfg = {}
    model_name = cfg.get("model", {}).get("name")

    # Auto detect peft checkpoint
    is_peft = False
    try:
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        is_peft = True

        if model_name is None:
            model_name = peft_config.base_model_name_or_path
        print("Detected PEFT adapter")
        print(f"Base model: {model_name}")
    except Exception:
        pass

    # HF model loading
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name if is_peft else checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name if is_peft else checkpoint_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Attach PEFT adapter if needed
    if is_peft:
        print(f"Loading PEFT adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=True,
        )
        model.print_trainable_parameters()
    return model, tokenizer, model_name
