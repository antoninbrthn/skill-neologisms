"""
Training script for baseline PEFT models on k-operation tasks. Evaluates at end of each epoch on:
  - Test sets (k-op combinations that include test_op)
  - Val sets for each operation count (1/2/3-ops)

Usage:
    PYTHONPATH=. python sequence_map_experiment/train_baselines.py --config_name baseline_lora.yaml
"""

import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf
from datasets import Dataset
from hydra import compose, initialize_config_dir

from src.config import PROJECT_ROOT
from src.peft_utils import create_peft_model, save_peft_adapter
from src.wandb_utils import init_wandb
from sequence_map_experiment.config import OUTPUT_DIR
from sequence_map_experiment.data import (
    split_on_output,
    generate_test_datasets,
    generate_train_val_datasets,
)
from sequence_map_experiment.model import load_any_model
from sequence_map_experiment.train_utils import KOpEvalCallback
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline PEFT models on k-operation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config file name (e.g., qwen_lora.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help='Config overrides in key=value format (e.g., dataset.test_op="[ADD]" dataset.max_ops=3)',
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Config: {args.config_name}")
    print(f"  Overrides: {args.overrides}")

    # Load config
    print(f"\nLoading config from {args.config_name}...")
    config_dir = os.path.join(PROJECT_ROOT, "sequence_map_experiment", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    print(f"\nConfiguration loaded.")

    # Extract required parameters
    skill_op = cfg.dataset.skill_op
    test_op = cfg.dataset.test_op

    if test_op is None:
        raise ValueError('Must specify dataset.test_op in config or via override (e.g., dataset.test_op="[ADD]")')

    print(f"\nKey parameters:")
    print(f"  Skill op: {skill_op}")
    print(f"  Test op: {test_op}")

    # Initialize WandB for this run
    max_ops = cfg.dataset.get("max_ops", 2)
    peft_type = cfg.peft.type if "peft" in cfg else "unknown"
    config_str = args.config_name.replace(".yaml", "") + f"_{max_ops}op_{test_op.replace('[', '').replace(']', '')}"
    wandb_run = init_wandb(
        cfg,
        run_formula=f"baseline_{peft_type}_{max_ops}op_{config_str}" + "_{timestamp}",
    )

    # Load base model
    pretrained_path = cfg.pretrained.checkpoint
    # Prefer a local checkpoint under OUTPUT_DIR if it exists, otherwise treat as HF repo id
    local_model_path = os.path.join(OUTPUT_DIR, pretrained_path)
    if os.path.exists(local_model_path):
        model_dir = local_model_path
    else:
        model_dir = pretrained_path
    print(f"\nLoading base model from {model_dir}...")
    model, tokenizer, model_name = load_any_model(model_dir, cfg)
    output_token = "="

    # Create PEFT model
    print(f"\nCreating PEFT model...")
    peft_checkpoint = cfg.peft.get("checkpoint_path", None) if "peft" in cfg else None
    if peft_checkpoint is not None:
        print(f"  PEFT checkpoint specified: {peft_checkpoint}")

    peft_model = create_peft_model(
        model=model,
        peft_cfg=cfg.peft,
        checkpoint_path=peft_checkpoint,
        tokenizer=tokenizer,
    )

    # Generate datasets
    num_train_samples = cfg.dataset.get("num_samples", 100_000)
    num_val_samples = cfg.dataset.get("val_dataset_size", 1000)
    num_test_samples = cfg.dataset.get("test_dataset_size", 200)
    max_ops = cfg.dataset.get("max_ops", 2)

    train_data_dict, val_data_dict = generate_train_val_datasets(
        cfg,
        skill_op,
        test_op,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        output_token=output_token,
    )

    # Combine all training and validation data
    train_data = []
    val_data = []
    for num_ops in range(1, max_ops + 1):
        train_data.extend(train_data_dict[num_ops])
        val_data.extend(val_data_dict[num_ops])

    print(f"\nDataset sizes:")
    for num_ops in range(1, max_ops + 1):
        print(f"  Train ({num_ops}-op): {len(train_data_dict[num_ops])}")
        print(f"     Examples: {train_data_dict[num_ops][:2]}")
    print(f"  Train (total): {len(train_data)}")
    for num_ops in range(1, max_ops + 1):
        print(f"  Val ({num_ops}-op): {len(val_data_dict[num_ops])}")
        print(f"     Examples: {val_data_dict[num_ops][:2]}")
    print(f"  Val (total): {len(val_data)}")

    # Generate test datasets
    test_datasets = generate_test_datasets(cfg, skill_op, test_op, num_samples=num_test_samples, output_token=output_token)

    # Split into prompts and labels
    train_prompts, train_labels = split_on_output(train_data, output_token=output_token)
    val_prompts, val_labels = split_on_output(val_data, output_token=output_token)

    # Create HF datasets
    train_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(train_prompts, train_labels)])
    val_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(val_prompts, val_labels)])

    print("\nFirst 3 training samples:")
    print("  " + "\n  ".join(train_data[:3]))

    # Setup training
    trainer_args_dict = OmegaConf.to_container(cfg.trainer_args, resolve=True)
    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        OUTPUT_DIR,
        cfg.output.save_dir,
        (ts_str + f"_{wandb_run.name.replace('/', '-')}" if cfg.wandb.enabled else ts_str),
    )

    trainer_args_dict.update(
        {
            "output_dir": output_dir,
            "report_to": "wandb" if cfg.wandb.enabled else None,
            "completion_only_loss": True,
            "save_strategy": "no",
        }
    )

    training_args = SFTConfig(**trainer_args_dict)

    print(f"\nTraining arguments configured:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output dir: {training_args.output_dir}")

    # Create trainer
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Add evaluation callback (use shared KOpEvalCallback with a small wrapper)
    class ModelWrapper:
        def __init__(self, model, tokenizer, output_token="="):
            self.model = model
            self.tokenizer = tokenizer
            self.output_token = output_token

    wrapper = ModelWrapper(peft_model, tokenizer, output_token=output_token)
    prompt_format_type = "id"

    eval_callback = KOpEvalCallback(
        adapter=wrapper,
        cfg=cfg,
        skill_op=skill_op,
        test_op=test_op,
        test_datasets=test_datasets,
        val_data_dict=val_data_dict,
        max_new_tokens=cfg.training.max_new_tokens,
        prefix_name=None,
        prompt_format_type=prompt_format_type,
    )
    trainer.add_callback(eval_callback)

    # Initial evaluation
    if cfg.trainer_args.get("eval_on_start", False):
        print(f"\n{'='*80}")
        print("Initial Evaluation (Before Training)")
        print(f"{'='*80}")
        wrapper.model = trainer.model
        eval_callback.run_evaluation(epoch=0)

    # Train
    print(f"\n{'='*80}")
    print("Training with HuggingFace Trainer...")
    print(f"{'='*80}\n")

    train_result = trainer.train()

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    print(f"Train metrics: {train_result.metrics}")

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    save_path = os.path.join(output_dir, "peft_adapter")
    save_peft_adapter(peft_model, save_path)
    print(f"Saved PEFT adapter to: {save_path}")

    if wandb_run is not None:
        wandb_run.summary["peft_adapter_path"] = save_path
        wandb_run.finish()
        print(f"WandB run finished: {wandb_run.url}")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
