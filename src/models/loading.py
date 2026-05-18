"""
Unified model loading utilities.

All HF model + tokenizer creation should go through ``load_base_hf_model``
(or the higher-level ``load_hf_model`` / ``load_any_model`` wrappers).
"""

import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    FineGrainedFP8Config,
)


# Low-level: load HF model + tokenizer (no skill / PEFT logic)
def load_base_hf_model(
    model_name: str,
    device: str = None,
    torch_dtype=torch.bfloat16,
    eval_mode: bool = True,
    pad_token: str = "<|pad|>",
    resize_embeds_for_pad: bool = True,
):
    """Single source-of-truth for loading an HF model + tokenizer.

    Handles architecture-specific quirks:
    - **Mistral** instruct variants: loaded via ``AutoModelForImageTextToText``
      with ``FineGrainedFP8Config(dequantize=True)``.
    - Everything else: ``AutoModelForCausalLM`` with the given *torch_dtype*.

    Returns:
        ``(model, tokenizer, device)``
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer and model for {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")

    if "mistralai" in model_name:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True),  # to use with BF16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    # Pad token
    if tokenizer.pad_token is None:
        if pad_token == "eos":
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            print(f"Tokenizer has no pad_token; adding one as {pad_token}")
            tokenizer.add_special_tokens({"pad_token": pad_token})
            if resize_embeds_for_pad:
                model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    if eval_mode:
        model.eval()

    return model, tokenizer, device


def get_embedding_weights(model) -> torch.nn.Parameter:
    """Return the input-embedding weight tensor for *model*.

    Prefer the public HF ``get_input_embeddings()`` API, then fall back to
    a few common nested layouts.
    """
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings.weight

    # eg gemma3 and mistral models use `inner.language_model.embed_tokens.weight`
    # qwen/llama use inner.embed_tokens.weight
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "get_input_embeddings"):
            embeddings = inner.get_input_embeddings()
            if embeddings is not None:
                return embeddings.weight

        for attr_path in (
            ("language_model", "embed_tokens"),
            ("model", "embed_tokens"),
            ("embed_tokens",),
        ):
            current = inner
            try:
                for attr in attr_path:
                    current = getattr(current, attr)
                return current.weight
            except AttributeError:
                continue

    raise AttributeError(f"Could not locate input embeddings for model of type {type(model).__name__}")


# High-level: config-based loaders
def infer_model_type(model_config):
    if "gpt" in model_config.name.lower():
        return "api"  # OpenAI API model
    else:
        return "hf"


def load_any_model(model_cfg):
    """Top-level dispatcher: API / HF / skill-token / PEFT models."""
    model_type = model_cfg.get("type", None)
    if model_type is None:
        model_type = infer_model_type(model_cfg)

    if model_type == "api":
        from src.models.api_models import load_api_model

        return load_api_model(model_cfg), None, None, None
    elif model_type == "peft":
        return load_peft_model(model_cfg)
    elif model_type in ["hf", "skill"]:
        return load_hf_model(model_cfg)
    else:
        raise ValueError(f"Invalid model type '{model_type}'. Choose 'hf', 'skill', 'peft', or 'api'.")


def load_hf_model(model_cfg):
    """Load model + tokenizer based on config. Returns (model, tokenizer, device, adapter_or_None)."""
    from src.models.skill_token_model import (
        SkillTokenModel,
    )  # deferred to avoid circular import

    model_name = model_cfg.name
    model_type = model_cfg.get("type", "hf")

    assert model_type in [
        "hf",
        "skill",
    ], f"Only 'hf' and 'skill' model types supported, got {model_type}"

    skills = model_cfg.get("skills", [])
    has_skills = (model_type == "skill") and len(skills) > 0

    if has_skills:
        # load base model, then attach skill tokens
        adapter = SkillTokenModel(model_name=model_name)
        for skill in skills:
            cp = skill.get("checkpoint_path")
            if cp:
                adapter.load_skills(cp, overwrite_existing=True)
                print(f"Loaded skill '{skill['name']}' from checkpoint: {cp}")
            else:
                adapter.create_skill(skill_id=skill.name, length=skill.length)
                print(f"Created new skill '{skill['name']}' with length {skill['length']} (no checkpoint provided)")
        adapter.set_skill_unembed_to_zero(verbose=True)
        return adapter.model, adapter.tokenizer, adapter.device, adapter
    else:
        model, tokenizer, device = load_base_hf_model(model_name, eval_mode=True)
        return model, tokenizer, device, None


def load_peft_model(model_cfg):
    """Load base model + PEFT adapter for inference. Returns (model, tokenizer, device, None).

    Expects ``model_cfg`` to contain:
    - ``name``: HuggingFace model name (e.g. ``google/gemma-3-4b-it``)
    - ``peft_adapter_path``: path to saved PEFT adapter directory

    For LoRA adapters the weights are merged into the base model and the
    PEFT wrapper is removed (``merge_and_unload``).  For prompt-tuning /
    prefix-tuning adapters merging is not applicable, so the PeftModel
    wrapper is kept as-is.
    """
    import json
    from peft import PeftModel as _PeftModel  # deferred to keep top-level imports light

    model_name = model_cfg.name
    adapter_path = model_cfg.get("peft_adapter_path", None)
    if adapter_path is None:
        raise ValueError("model.peft_adapter_path is required when model.type='peft'")

    # Detect adapter type from saved config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    peft_type = None
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            peft_type = json.load(f).get("peft_type", "").lower()
        print(f"Detected PEFT adapter type: {peft_type}")

    print(f"\nLoading base model: {model_name}")
    model, tokenizer, device = load_base_hf_model(model_name, eval_mode=True)

    print(f"Loading PEFT adapter from: {adapter_path}")
    model = _PeftModel.from_pretrained(model, adapter_path)

    # Only LoRA adapters can be merged into the base weights
    if peft_type in ["lora"]:
        model = model.merge_and_unload()
        print("PEFT adapter merged and unloaded for inference.")
    else:
        print(f"Keeping PeftModel wrapper for {peft_type} adapter (not mergeable).")

    model.eval()
    return model, tokenizer, device, None
