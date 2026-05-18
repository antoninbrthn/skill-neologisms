"""
Generation utilities for SkillMix evaluation.
Handles loading HF models (optionally with skill tokens) and batched text generation.
"""

import torch
from typing import List, Dict, Optional
import tqdm
from src.models.api_models import OpenAIModel
from src.models.skill_token_model import SkillTokenModel
from src.models.loading import load_hf_model


def load_model(cfg):
    """Load model + tokenizer based on config. Returns (model, tokenizer, device, adapter_or_None).

    Delegates to the canonical ``load_hf_model`` in ``src.models.loading``.
    """
    return load_hf_model(cfg.model)


@torch.inference_mode()
def generate_batch_hf(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    do_sample: bool = True,
    batch_size: int = 8,
    adapter: Optional[SkillTokenModel] = None,
    **kwargs,
) -> List[str]:
    """Generate completions for a list of prompts using HF generate, in batches."""
    all_outputs = []
    # for i in range(0, len(prompts), batch_size):
    for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
        batch = prompts[i : i + batch_size]
        # inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        # Adding the below to avoid this error:
        # | ValueError: The following `model_kwargs` are not used by the model: ['token_type_ids'] (note: typos in the generate arguments will also show up in this list)
        inputs = tokenizer(
            batch,
            return_token_type_ids=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample and temperature > 1e-5,
            temperature=temperature if (do_sample and temperature > 1e-5) else 1.0,
        )
        output_ids = model.generate(**inputs, **gen_kwargs)
        # strip prompt tokens
        prompt_len = inputs["input_ids"].shape[1]
        for j in range(output_ids.shape[0]):
            gen_ids = output_ids[j, prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_outputs.append(text)

    return all_outputs


def generate_batch_gpt(
    model: OpenAIModel,
    prompts: List[str],
    batch_size: int = 8,
) -> List[str]:
    """Send grading prompts to the API model and return raw grading outputs.

    Args:
        model: The grading model
        prompts: List of prompts to grade
        batch_size: Number of prompts to process per batch

    Returns:
        List of raw grading outputs
    """
    if batch_size <= 0 or batch_size >= len(prompts):
        return model._generate(input_text=prompts)

    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = model._generate(input_text=batch)
        if isinstance(outputs, str):
            outputs = [outputs]
        all_outputs.extend(outputs)

    return all_outputs


def extract_answer(text: str) -> str:
    """Extract the answer portion from a generation (text after 'Answer:')."""
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    if "Explanation" in text:
        text = text.split("Explanation")[0]
    return text.strip()


if __name__ == "__main__":
    # Quick demo: load a small model and generate from a test prompt
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"name": "Qwen/Qwen2.5-0.5B", "type": "hf", "skills": []},
        }
    )
    model, tokenizer, device, adapter = load_model(cfg)
    prompts = ["Hello, how are you?", "Tell me a joke."]
    outputs = generate_batch_hf(model, tokenizer, prompts, device, max_new_tokens=50, batch_size=2)
    for p, o in zip(prompts, outputs):
        print(f"Prompt: {p}\nOutput: {o}\n")
