"""
Finetune a huggingface model on the digit-sequence task.
"""

import itertools
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict
import numpy as np
import re
from omegaconf import OmegaConf

from sequence_map_experiment.config import OUTPUT_DIR
from sequence_map_experiment.data import (
    SequenceTaskDataset,
    generate_sample_data,
    split_on_output,
)
from sequence_map_experiment.utils import (
    load_pretraining_config,
    train_model,
    evaluate_model,
)


def merge_training_configs(default_config: Dict, phase_config: Dict) -> Dict:
    """Merge phase-specific training config with default training config.

    Args:
        default_config: Default training configuration
        phase_config: Phase-specific training overrides

    Returns:
        Merged configuration dictionary
    """
    merged = dict(default_config)
    if phase_config:
        merged.update(phase_config)
    return merged


def train_over_phases(
    model,
    tokenizer,
    phases,
    output_dir_run,
    default_training_config,
    wandb_run,
    device="cuda",
    report_to="wandb",
    task_token_length=1,
    output_token="[OUTPUT]",
):
    """
    Main training loop over multiple phases.

    :param model: Model to train
    :param tokenizer: Tokenizer to use
    :param phases: List of phase configurations
    :param output_dir_run: Output directory for the run
    :param device: Device to use for training
    :param report_to: Reporting method for Trainer ('wandb' or 'none')
    :param task_token_length: Length of task tokens to use. If 1, will use task tokens as is (eg "[ADD]"). Otherwise, expand to eg: '<|[ASC]-1|>', '<|[ASC]-2|>', ...
    """
    # Iterate over phases
    for phase_idx, phase_cfg in enumerate(phases):
        phase_name = phase_cfg.get("name", f"phase-{phase_idx+1}")
        should_train = phase_cfg.get("train", True)

        phase_dir = os.path.join(output_dir_run, phase_name)

        print("\n" + "=" * 80)
        print(
            f"{phase_name.upper()}: {phase_cfg.get('dataset', {}).get('min_ops', 1)}-{phase_cfg.get('dataset', {}).get('max_ops', 1)} operation tasks"
        )
        print("=" * 80)

        if should_train:
            # Get dataset config
            dataset_cfg = phase_cfg.get("dataset", {})

            # Generate training data
            print(f"\nGenerating {phase_name} data...")
            ops = dataset_cfg.get("ops", None)
            seq_len = dataset_cfg.get("seq_len", 8)  # default
            seq_len_min = dataset_cfg.get("seq_len_min", seq_len)
            reject_len = dataset_cfg.get("reject_len", None)
            reject_ops = dataset_cfg.get("reject_ops", [])

            # Handle reject_2ops_ratio if present
            reject_2ops_ratio = dataset_cfg.get("reject_2ops_ratio", None)
            if reject_2ops_ratio is not None and ops is not None:
                num_reject = int(len(list(itertools.permutations(ops, 2))) * reject_2ops_ratio)
                print(f"Rejecting {num_reject} 2-op combinations from training data")
                for _ in range(num_reject):
                    reject_ops.append("".join(np.random.choice(ops, 2, replace=False)))

            # Handle reject_3ops_ratio if present
            reject_3ops_ratio = dataset_cfg.get("reject_3ops_ratio", None)
            if reject_3ops_ratio is not None and ops is not None:
                num_reject = int(len(list(itertools.permutations(ops, 3))) * reject_3ops_ratio)
                print(f"Rejecting {num_reject} 3-op combinations from training data")
                for _ in range(num_reject):
                    reject_ops.append("".join(np.random.choice(ops, 3, replace=False)))

            # Log to wandb
            if wandb_run is not None:
                wandb_run.config.update({f"{phase_name}_reject_ops": reject_ops})

            print("Using ops:", ops)
            print("Using seq_len:", seq_len)
            print("Using seq_len_min:", seq_len_min)
            print("Using reject_len:", reject_len)
            print("Using reject_ops:", reject_ops)

            train_data = generate_sample_data(
                dataset_cfg.get("num_train_samples", 10000),
                min_ops=dataset_cfg.get("min_ops", 1),
                max_ops=dataset_cfg.get("max_ops", 1),
                ops=ops,
                seq_len=(seq_len_min, seq_len),
                reject_len=reject_len,
                reject_ops=reject_ops,
                task_token_length=task_token_length,
                output_token=output_token,
            )

            test_data = generate_sample_data(
                dataset_cfg.get("num_test_samples", 500),
                min_ops=dataset_cfg.get("min_ops", 1),
                max_ops=dataset_cfg.get("max_ops", 1),
                ops=ops,
                seq_len=(seq_len_min, seq_len),
                reject_len=reject_len,
                reject_ops=reject_ops,
                task_token_length=task_token_length,
                output_token=output_token,
            )
            print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")

            # Create datasets
            print(f"Creating datasets for {phase_name}...")
            random_left_pad = dataset_cfg.get("random_left_pad", False)
            random_skill_pad = dataset_cfg.get("random_skill_pad", False)
            train_dataset = SequenceTaskDataset(
                train_data,
                tokenizer,
                random_left_pad=random_left_pad,
                random_skill_pad=random_skill_pad,
                output_token=output_token,
            )
            test_dataset = SequenceTaskDataset(
                test_data,
                tokenizer,
                random_left_pad=random_left_pad,
                random_skill_pad=random_skill_pad,
                output_token=output_token,
            )

            print("First 5 training samples:", train_data[:5], train_dataset[:5])
            print("First 5 test samples:", test_data[:5], test_dataset[:5])

            # Merge training configs
            phase_training_cfg = phase_cfg.get("training", {})
            training_config = merge_training_configs(
                default_training_config,
                (OmegaConf.to_container(phase_training_cfg, resolve=True) if phase_training_cfg else {}),
            )

            # Train model
            print(f"\nStarting {phase_name} training...")
            print(f"Training config: {training_config}")

            trainer = train_model(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                output_dir=phase_dir,
                report_to=report_to,
                **training_config,
            )

            # Evaluate on test set
            print(f"\nEvaluating {phase_name} on test set...")
            evaluate_model(
                model,
                tokenizer,
                test_data[:100],
                device=device,
                output_token=output_token,
            )
            print(f"\n✓ {phase_name} training complete!")
            print(f"Model saved to: {phase_dir}/final_model")
        else:
            # Load existing model
            print(f"\nSkipping {phase_name} training, loading existing model...")
            model_path = f"{phase_dir}/final_model"
            if os.path.exists(model_path):
                # Check if model already has LoRA adapters
                if isinstance(model, PeftModel):
                    # Model already has LoRA, unload before loading new checkpoint
                    print("Unloading existing LoRA adapters...")
                    model = model.unload()

                # Load the checkpoint (this will add LoRA adapters)
                model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                print(f"Loaded model from: {model_path}")
                model.print_trainable_parameters()
            else:
                print(f"Warning: Model not found at {model_path}, continuing with current model")

    print("\n" + "=" * 80)
    print("Training pipeline complete!")
    print("=" * 80)


def main(config_name: str = None, overrides: list = []):
    """Main training script using configuration.

    Args:
        config_name: Name of config file (e.g., 'default.yaml'). Defaults to 'default.yaml'
        overrides: List of config overrides in key=value format
    """
    # Load configuration
    if config_name is None:
        config_name = "default.yaml"

    print(f"Loading config: {config_name}")
    if overrides:
        print(f"With overrides: {overrides}")
    cfg = load_pretraining_config(config_name, overrides)

    # Determine output directory
    base_output_dir = cfg.output_dir if cfg.output_dir else OUTPUT_DIR

    # Determine run ID
    if cfg.run_id is None:
        run_id = 0
        if os.path.exists(base_output_dir):
            for dirname in os.listdir(base_output_dir):
                if re.match(r"run\d+", dirname):
                    run_num = int(dirname.replace("run", ""))
                    if run_num >= run_id:
                        run_id = run_num + 1
    else:
        run_id = cfg.run_id

    print(f"\nRun ID: {run_id}")
    output_dir_run = os.path.join(base_output_dir, f"run{run_id}")

    # Initialize W&B if enabled
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        from datetime import datetime

        wandb_name = cfg.wandb.get("name") or f"pretrain_qwen_run{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=wandb_name,
            tags=list(cfg.wandb.get("tags", [])),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"W&B run initialized: {wandb_name}")

    # Load pretrained model and make adapter
    print(f"Loading model {cfg.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_cfg.lora_r,
        lora_alpha=cfg.lora_cfg.lora_alpha,
        lora_dropout=cfg.lora_cfg.lora_dropout,
        target_modules=list(cfg.lora_cfg.target_modules),
    )

    # Check if we need to skip phase 1 and load from checkpoint
    phases = cfg.get("phases", [])
    skip_phase_1 = phases and len(phases) > 0 and not phases[0].get("train", True)

    if skip_phase_1:
        # Don't apply LoRA yet - we will load it from checkpoint
        print("Phase 1 training is skipped - will load LoRA from checkpoint")
    else:
        # Apply LoRA for fresh training
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Get default training config
    default_training_config = OmegaConf.to_container(cfg.get("training", {}), resolve=True) if cfg.get("training") else {}

    # Iterate over all configured phases
    phases = cfg.get("phases", [])
    if not phases:
        print("Warning: No phases defined in config!")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(phases)} phases in configuration")
    print(f"{'='*80}")

    # print a quick recap of phases
    for phase_idx, phase_cfg in enumerate(phases):
        phase_name = phase_cfg.get("name", f"phase-{phase_idx+1}")
        should_train = phase_cfg.get("train", True)
        print(f"Phase {phase_idx+1}: {phase_name} - {'TRAIN' if should_train else 'SKIP TRAINING'}")
        print(f"  Dataset config: {phase_cfg.get('dataset', {})}")
        print(f"  Training config: {phase_cfg.get('training', {})}")

    train_over_phases(
        model,
        tokenizer,
        phases,
        output_dir_run,
        default_training_config,
        wandb_run,
        output_token="=",
    )

    # Finish W&B run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import sys

    config_name = sys.argv[1] if len(sys.argv) > 1 else None
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    main(config_name, overrides)
