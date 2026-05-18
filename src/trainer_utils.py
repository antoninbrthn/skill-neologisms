"""Trainer and evaluation utilities for skill neologisms training."""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from trl import SFTTrainer
from transformers import TrainerCallback

from src.models.skill_token_model import SkillTokenModel
from src.renderer.prompts_renderer import PromptLibrary


def generate_prompt_dataset(
    data_generator: Callable,
    prompt_library: PromptLibrary,
    prompt_name: str,
    num_samples: int,
    data_generator_kwargs: Dict[str, Any] = None,
    num_examples_range: Tuple[int, int] = (1, 5),
    layout_id: int = None,
) -> List[Tuple[str, str]]:
    """Generate a list of (prompt, label) pairs from a task generator."""
    data_generator_kwargs = data_generator_kwargs or {}

    raw_samples = [data_generator(**data_generator_kwargs) for _ in range(num_samples)]
    queries = [{"x1": sample[0][0], "x2": sample[0][1], "y": sample[1]} for sample in raw_samples]

    example_samples = [data_generator(**data_generator_kwargs) for _ in range(num_samples * num_examples_range[1])]
    example_queries = [{"x1": sample[0][0], "x2": sample[0][1], "y": sample[1]} for sample in example_samples]

    rendered = []
    for q in queries:
        context = q.copy()
        num_examples = np.random.randint(num_examples_range[0], num_examples_range[1] + 1)
        context["examples"] = np.random.choice(example_queries, size=num_examples, replace=False)
        prompt_str = prompt_library.render(name=prompt_name, context=context, layout_id=layout_id)
        rendered.append((prompt_str, q["y"]))

    return rendered


class SkillTokenTrainer(SFTTrainer):
    """SFT trainer with gradient masking to update only active skill-token rows."""

    def __init__(self, adapter: SkillTokenModel, *args, **kwargs):
        self.adapter = adapter
        super().__init__(*args, **kwargs)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            self.adapter.zero_out_non_skill_grads(input_ids)

        return loss.detach()

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.adapter.get_optimizer(lr=self.args.learning_rate)
        return self.optimizer


def evaluate_with_generation(
    adapter: SkillTokenModel,
    prompts: List[str],
    labels: List[str],
    skill_id: str,
    batch_size: int = 8,
    max_new_tokens: int = 10,
) -> Tuple[float, List[str], List[str]]:
    """Evaluate by generation and exact-match comparison."""
    from tqdm import tqdm

    adapter.model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
            batch_prompts = prompts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            full_prompts = [adapter._expand_skill_tokens(p) for p in batch_prompts]

            prompt_encodings = adapter.tokenizer(
                full_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            prompt_inputs = {
                "input_ids": prompt_encodings["input_ids"].to(adapter.model.device),
                "attention_mask": prompt_encodings["attention_mask"].to(adapter.model.device),
            }

            outputs = adapter.model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=adapter.tokenizer.pad_token_id,
                do_sample=False,
            )

            input_length = prompt_encodings["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            predictions = adapter.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for pred, label in zip(predictions, batch_labels):
                pred = pred[: len(label)]
                all_predictions.append(pred)
                all_labels.append(label)

                if pred == label:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    adapter.model.train()

    return accuracy, all_predictions, all_labels
