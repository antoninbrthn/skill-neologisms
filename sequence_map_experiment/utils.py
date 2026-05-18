import os
import torch
from transformers import Trainer, TrainingArguments
from typing import List, Optional
import numpy as np
from hydra import compose, initialize_config_dir


from src.config import PROJECT_ROOT


def load_pretraining_config(config_name: str, overrides: list = []):
    """Load configuration using Hydra.

    Args:
        config_name: Name of config file (e.g., 'default.yaml')
        overrides: List of config overrides in key=value format

    Returns:
        OmegaConf DictConfig object
    """
    config_dir = os.path.join(PROJECT_ROOT, "sequence_map_experiment", "pretraining_config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def load_cfg_from_dict(cfg):
    """Unflatten a flat dict with dot-separated keys into a nested OmegaConf DictConfig."""
    from omegaconf import OmegaConf

    unflattened_cfg = {}
    for key, value in cfg.items():
        if "." in key:
            main_key, sub_key = key.split(".", 1)
            if main_key not in unflattened_cfg:
                unflattened_cfg[main_key] = {}
            unflattened_cfg[main_key][sub_key] = value
        else:
            unflattened_cfg[key] = value
    cfg = OmegaConf.create(unflattened_cfg)
    return cfg


# Training
def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str = "./checkpoints",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    report_to: str = "none",
    **kwargs,
):
    """Train the model using HuggingFace Trainer."""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=report_to,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {output_dir}/final_model")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    return trainer


# Evaluation
def evaluate_model(
    model,
    tokenizer,
    test_data: List[str],
    device="cuda",
    verbose=True,
    prevent_batch=False,
    output_token: str = "=",
):
    """Evaluate model on test data."""
    if not prevent_batch:
        return evaluate_model_batch(
            model,
            tokenizer,
            test_data,
            device=device,
            verbose=verbose,
            output_token=output_token,
        )

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    detailed = []
    for i, example in enumerate(test_data):
        if output_token not in example:
            raise ValueError(f"Test example does not contain output token '{output_token}': {example}")
        input_part, expected_output = split_on_last_output_token(example, output_token)
        prompt = input_part + output_token
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=len(expected_output) + 5,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Extract generated output
        if output_token in generated:
            generated_output = generated.split(output_token)[1].replace("[PAD]", "").replace("[EOS]", "").strip()
        else:
            # generated_output = ""
            # Some models (eg PEFT) may just output the completion without the prompt.
            generated_output = generated.replace("[BOS]", "").replace("[PAD]", "").replace("[EOS]", "").strip()

        # Check correctness
        is_correct = generated_output[: len(expected_output)] == expected_output
        if is_correct:
            correct += 1
        total += 1
        detailed.append([generated_output, expected_output])

    accuracy = correct / total if total > 0 else 0

    if verbose:
        print(f"\nExample {i+1}:")
        print(f"  Input:    {input_part}")
        print(f"  Expected: {expected_output}")
        print(f"  Got:      {generated_output}")
        print(f"  Status:   {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        print(f"\n{'='*80}")
        print(f"Accuracy on {len(test_data)} examples: {accuracy:.2%} ({correct}/{total})")
        print(f"{'='*80}\n")

    return accuracy, detailed


def evaluate_model_batch(
    model,
    tokenizer,
    test_data: List[str],
    device="cuda",
    verbose=True,
    batch_size: int = 32,
    output_token: str = "=",
):
    """Evaluate model on test data using batched generation."""

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    detailed = []

    assert np.all([output_token in ex for ex in test_data]), f"All test examples must contain {output_token} token."

    # Process in batches
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_examples = test_data[batch_start:batch_end]

        # Prepare batch data
        batch_prompts = []
        batch_expected_outputs = []
        batch_input_parts = []

        for example in batch_examples:
            input_part, expected_output = split_on_last_output_token(example, output_token)
            prompt = input_part + output_token
            batch_prompts.append(prompt)
            batch_expected_outputs.append(expected_output)
            batch_input_parts.append(input_part)

        # Tokenize batch with padding
        max_output_len = max(len(exp) for exp in batch_expected_outputs)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(device)

        # Generate for batch
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_output_len + 5,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and evaluate batch
        for i, (output_id, expected_output, input_part) in enumerate(zip(output_ids, batch_expected_outputs, batch_input_parts)):
            generated = tokenizer.decode(output_id, skip_special_tokens=False)

            # Extract generated output
            if output_token in generated:
                generated_output = generated.split(output_token)[1].replace("[PAD]", "").replace("[EOS]", "").strip()
            else:
                # Some models (eg PEFT) may just output the completion without the prompt.
                generated_output = generated.replace("[BOS]", "").replace("[PAD]", "").replace("[EOS]", "").strip()

            # Check correctness
            is_correct = generated_output[: len(expected_output)] == expected_output
            if is_correct:
                correct += 1
            total += 1
            detailed.append([generated_output, expected_output])

            # Store last example for verbose output
            if verbose and i == len(batch_examples) - 1 and batch_end == len(test_data):
                last_input = input_part
                last_expected = expected_output
                last_generated = generated_output
                last_correct = is_correct

    accuracy = correct / total if total > 0 else 0

    if verbose and total > 0:
        print(f"\nLast Example:")
        print(f"  Input:    {last_input}")
        print(f"  Expected: {last_expected}")
        print(f"  Got:      {last_generated}")
        print(f"  Status:   {'✓ CORRECT' if last_correct else '✗ WRONG'}")

        print(f"\n{'='*80}")
        print(f"Accuracy on {total} examples: {accuracy:.2%} ({correct}/{total})")
        print(f"{'='*80}\n")

    return accuracy, detailed


def split_on_last_output_token(example: str, output_token: str):
    """Split example on the last occurrence of output_token."""
    if output_token not in example:
        raise ValueError(f"Test example does not contain output token '{output_token}': {example}")
    # check if multiple output tokens are present
    if example.count(output_token) > 1:
        # if so, split on the last occurrence
        split_pos = example.rfind(output_token)
        input_part = example[:split_pos]
        expected_output = example[split_pos + len(output_token) :]
    else:
        input_part, expected_output = example.split(output_token)
    return input_part, expected_output


def evaluate_model_kpass(
    model,
    tokenizer,
    test_data: List[str],
    k: int = 5,
    device="cuda",
    verbose=True,
    batch_size: Optional[int] = None,
    temperature=1,
    output_token: str = "=",
):
    """Evaluate model on test data with k completions per instance.

    Computes pass@i accuracy for i=1..k, where pass@i is the percentage of
    examples where at least one of the first i completions is correct.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_data: List of test examples in format "input[OUTPUT]expected_output"
        k: Number of completions to generate per instance
        device: Device to run on
        verbose: Whether to print detailed results
        batch_size: If provided, generate k completions in batches for efficiency.
                   If None, generates all k completions sequentially.
        temperature: Sampling temperature.
        output_token: String that separates input from expected output in test examples

    Returns:
        dict: Contains 'pass_at_k' (dict mapping i to pass@i accuracy) and
              'detailed' (list of [k_generated_outputs, expected_output] per example)
    """
    model.eval()
    model.to(device)

    # Results for each example: list of (expected_output, [k completions])
    all_results = []

    for example_idx, example in enumerate(test_data):

        input_part, expected_output = split_on_last_output_token(example, output_token)
        prompt = input_part + output_token

        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate k completions
        completions = []

        if batch_size is not None and batch_size > 1:
            # Batch generation: generate multiple completions at once
            num_batches = (k + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                batch_k = min(batch_size, k - batch_idx * batch_size)

                # Replicate input for batch generation
                batch_input_ids = input_ids.repeat(batch_k, 1)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=batch_input_ids,
                        max_new_tokens=len(expected_output) + 5,
                        num_beams=1,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode each completion in the batch
                for i in range(batch_k):
                    generated = tokenizer.decode(output_ids[i], skip_special_tokens=False)

                    # Extract generated output
                    if output_token in generated:
                        generated_output = generated.split(output_token)[1].replace("[PAD]", "").replace("[EOS]", "").strip()
                    else:
                        generated_output = generated.replace("[BOS]", "").replace("[PAD]", "").replace("[EOS]", "").strip()

                    completions.append(generated_output)
        else:
            # Sequential generation: generate one at a time
            for _ in range(k):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=len(expected_output) + 5,
                        num_beams=1,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)

                # Extract generated output
                if output_token in generated:
                    generated_output = generated.split(output_token)[1].replace("[PAD]", "").replace("[EOS]", "").strip()
                else:
                    generated_output = generated.replace("[BOS]", "").replace("[PAD]", "").replace("[EOS]", "").strip()

                completions.append(generated_output)

        all_results.append((expected_output, completions, input_part))

    # Compute pass@i for i=1..k
    pass_at_k = {}

    for i in range(1, k + 1):
        correct = 0
        for expected_output, completions, _ in all_results:
            # Check if any of the first i completions is correct
            is_correct = any(comp[: len(expected_output)] == expected_output for comp in completions[:i])
            if is_correct:
                correct += 1

        accuracy = correct / len(all_results) if len(all_results) > 0 else 0
        pass_at_k[i] = accuracy

    if verbose:
        print(f"\n{'='*80}")
        print(f"K-Pass Evaluation Results (k={k})")
        print(f"{'='*80}")

        # Show a few example completions
        num_examples_to_show = min(3, len(all_results))
        for idx in range(num_examples_to_show):
            expected_output, completions, input_part = all_results[idx]
            print(f"\nExample {idx+1}:")
            print(f"  Input:    {input_part}")
            print(f"  Expected: {expected_output}")
            for j, comp in enumerate(completions[: min(3, k)], 1):
                is_correct = comp[: len(expected_output)] == expected_output
                status = "✓" if is_correct else "✗"
                print(f"  Pass {j}:   {comp} {status}")
            if k > 3:
                print(f"  ... ({k - 3} more completions)")

        print(f"\n{'='*80}")
        print(f"Pass@i Accuracy on {len(all_results)} examples:")
        for i in range(1, k + 1):
            print(f"  pass@{i}: {pass_at_k[i]:.2%}")
        print(f"{'='*80}\n")

    # Prepare detailed output
    detailed = [[completions, expected_output] for expected_output, completions, _ in all_results]

    return {"pass_at_k": pass_at_k, "detailed": detailed}
