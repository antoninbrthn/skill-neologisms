"""
Training skill neologisms on k-operation tasks. Evaluates at end of each epoch on:
  - Test sets (k-op combinations that include test_op)
  - Val sets for each operation count (1/2/3-ops)

Usage:
    PYTHONPATH=. python sequence_map_experiment/train_neologisms.py --config_name skill_tokens.yaml
"""

import os
import random
import numpy as np
import argparse
from datetime import datetime
from omegaconf import OmegaConf, open_dict
from datasets import Dataset
from hydra import compose, initialize_config_dir
import wandb

from src.config import PROJECT_ROOT
from src.models.skill_token_model import load_skill_model_from_cfg
from src.trainer_utils import SkillTokenTrainer
from src.wandb_utils import init_wandb
from sequence_map_experiment.config import OUTPUT_DIR
from sequence_map_experiment.data import (
    PRETRAIN_OPS,
    generate_test_datasets,
    generate_train_val_datasets,
    split_on_output,
)
from sequence_map_experiment.train_utils import (
    KOpEvalCallback,
    expand_skill_token_texts,
)
from trl import SFTConfig


def load_from_wandb(run_id, project="skill-neologisms", entity=None):
    """
    Load config and checkpoint path from a WandB run.

    Args:
        run_id: WandB run ID or name
        project: WandB project name
        entity: WandB entity name

    Returns:
        tuple: (cfg, skill_checkpoint_path)
    """
    entity = entity or os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WandB entity is required. Set WANDB_ENTITY or pass entity=...")

    print(f"Loading from WandB run: {run_id}")
    print(f"  Project: {entity}/{project}")

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"  Run name: {run.name}")
    print(f"  Created: {run.created_at}")
    print(f"  State: {run.state}")

    # Get config from wandb
    wandb_config = run.config
    cfg = OmegaConf.create(wandb_config)

    # Get skill checkpoint path from summary
    if "skill_token_path" in run.summary:
        skill_checkpoint_path = run.summary["skill_token_path"]
        print(f"  Skill checkpoint from summary: {skill_checkpoint_path}")
    else:
        print("  Warning: No skill_token_path in run summary")
        skill_checkpoint_path = None

    return cfg, skill_checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Train skill tokens on k-operation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config file name (e.g., skill_tokens.yaml)",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        required=False,
        help="WandB run ID to load skill checkpoint from",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="skill-neologisms",
        help="WandB project name",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help='Config overrides in key=value format (e.g., dataset.test_op="[ADD]" dataset.max_ops=3)',
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Config: {args.config_name}")
    if args.wandb_id is not None:
        print(f"  WandB run ID: {args.wandb_id}")
    print(f"  Overrides: {args.overrides}")

    # Load from WandB
    if args.wandb_id is not None:
        wandb_cfg, skill_checkpoint_path = load_from_wandb(args.wandb_id)
    else:
        wandb_cfg = {}
        skill_checkpoint_path = None

    # Load new config and merge
    print(f"\nLoading config from {args.config_name}...")
    config_dir = os.path.join(PROJECT_ROOT, "sequence_map_experiment", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        new_cfg = compose(config_name=args.config_name, overrides=args.overrides)

    # Merge configs
    cfg = OmegaConf.merge(wandb_cfg, new_cfg)

    # Extract required parameters
    skill_op = cfg.dataset.skill_op if cfg.dataset.skill_op is not None else cfg.skill.name
    test_op = cfg.dataset.test_op
    # Add skill_checkpoint_path in cfg.skill if not already set
    if skill_checkpoint_path is not None:
        with open_dict(cfg):
            cfg.skill.checkpoint_path = skill_checkpoint_path
    elif cfg.skill.get("checkpoint_path", None) is not None:
        skill_checkpoint_path = cfg.skill.checkpoint_path
    else:
        with open_dict(cfg):
            cfg.skill.checkpoint_path = None

    if test_op is None:
        raise ValueError('Must specify dataset.test_op in config or via override (e.g., dataset.test_op="[ADD]")')

    print(f"\nKey parameters:")
    print(f"  Skill op: {skill_op}")
    print(f"  Test op: {test_op}")
    print(f"  Skill checkpoint: {skill_checkpoint_path}")

    # Sample other_ops if n_other_ops is specified
    # If other_ops is given in cfg: use for other_ops; otherwise use all PRETRAIN_OPS excluding test_op
    other_ops = cfg.dataset.get("other_ops", None)
    if other_ops is None:
        other_ops = [op for op in PRETRAIN_OPS if op != test_op]
    # If n_other_ops is given in cfg, sample that many from other_ops
    n_other_ops = cfg.dataset.get("n_other_ops", None)
    other_ops_perm_id = cfg.dataset.get("other_ops_perm_id", None)
    if n_other_ops is not None:
        if other_ops_perm_id is not None:
            from itertools import combinations

            combs = list(combinations(other_ops, n_other_ops))
            if other_ops_perm_id < 0 or other_ops_perm_id >= len(combs):
                raise ValueError(f"other_ops_perm_id {other_ops_perm_id} out of range (0 to {len(combs)-1}) for n_other_ops={n_other_ops}")
            print(f"Using other_ops_perm_id {other_ops_perm_id} to select other_ops from {other_ops}")
            other_ops = list(combs[other_ops_perm_id])
        else:
            print(f"Sampling {n_other_ops} other_ops from available {other_ops}")
            other_ops = random.sample(other_ops, n_other_ops)
    print(f"Using other_ops: {other_ops}")
    with open_dict(cfg):
        cfg.dataset.other_ops = other_ops
    print(" Updated cfg.dataset.other_ops.")

    # Initialize wandb
    max_ops = cfg.dataset.get("max_ops", 2)
    config_str = args.config_name.replace(".yaml", "") + f"_{max_ops}op_{test_op.replace('[', '').replace(']', '')}"
    wandb_run = init_wandb(cfg, run_formula=f"skill_{max_ops}op_{config_str}" + "_{timestamp}")
    wandb_run.config.update({"parent_wandb_id": args.wandb_id}, allow_val_change=True)

    # Load skill model adapter
    print(f"\nCreating SkillTokenModel adapter...")
    adapter = load_skill_model_from_cfg(cfg, skill_checkpoint_path=skill_checkpoint_path)
    if len(adapter.model_cfg) == 0:
        print("Warning: adapter.model_cfg is empty after loading skill model.")
    else:
        print(f"  Loaded adapter model_cfg: {adapter.model_cfg}")
    print(f"  Loaded skills: {list(adapter.skill_tokens.keys())}")

    # Generate datasets
    num_train_samples = cfg.dataset.get("num_samples", 100_000)
    num_val_samples = cfg.dataset.get("val_dataset_size", 1000)
    num_test_samples = cfg.dataset.get("test_dataset_size", 200)
    max_ops = cfg.dataset.get("max_ops", 2)
    task_token_length = adapter.model_cfg.get("adaptation", {}).get("task_length", 1)
    train_data_dict, val_data_dict = generate_train_val_datasets(
        cfg,
        skill_op,
        test_op,
        other_ops=other_ops,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        task_token_length=task_token_length,
    )

    # Combine all training and validation data
    train_data = []
    val_data = []
    for num_ops in train_data_dict.keys():
        train_data.extend(train_data_dict[num_ops])
        val_data.extend(val_data_dict[num_ops])

    print(f"\nDataset sizes:")
    for num_ops in train_data_dict.keys():
        print(f"  Train ({num_ops}-op): {len(train_data_dict[num_ops])}")
    print(f"  Train (total): {len(train_data)}")
    for num_ops in val_data_dict.keys():
        print(f"  Val ({num_ops}-op): {len(val_data_dict[num_ops])}")
    print(f"  Val (total): {len(val_data)}")

    # Generate test datasets
    test_datasets = generate_test_datasets(
        cfg,
        skill_op,
        test_op,
        num_samples=num_test_samples,
        task_token_length=task_token_length,
    )

    # Prepare data for trainer
    # Expand skill tokens in data
    train_data = expand_skill_token_texts(train_data, cfg.skill.name, adapter)
    val_data = expand_skill_token_texts(val_data, cfg.skill.name, adapter)

    # Expand validation datasets for each operation count
    for num_ops in val_data_dict.keys():
        val_data_dict[num_ops] = expand_skill_token_texts(val_data_dict[num_ops], cfg.skill.name, adapter)

    # Expand test datasets
    for num_ops in test_datasets.keys():
        for seq_len in test_datasets[num_ops]:
            for perm_key in test_datasets[num_ops][seq_len]:
                test_datasets[num_ops][seq_len][perm_key] = expand_skill_token_texts(
                    test_datasets[num_ops][seq_len][perm_key], cfg.skill.name, adapter
                )

    # Split into prompts and labels
    train_prompts, train_labels = split_on_output(train_data)
    val_prompts, val_labels = split_on_output(val_data)

    # Create HF datasets
    train_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(train_prompts, train_labels)])
    val_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(val_prompts, val_labels)])

    print("\nRandom 5 training samples:")
    print("  " + "\n  ".join(np.random.choice(train_data, size=5, replace=False)))
    print("\nRandom 5 validation samples:")
    print("  " + "\n  ".join(np.random.choice(val_data, size=5, replace=False)))

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
            "save_strategy": cfg.trainer_args.get("save_strategy", "no"),
        }
    )

    training_args = SFTConfig(**trainer_args_dict)

    print(f"\nTraining arguments configured:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output dir: {training_args.output_dir}")

    # Create trainer
    trainer = SkillTokenTrainer(
        adapter=adapter,
        model=adapter.model,
        processing_class=adapter.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Add evaluation callback
    eval_callback = KOpEvalCallback(
        adapter=adapter,
        cfg=cfg,
        skill_op=skill_op,
        test_op=test_op,
        test_datasets=test_datasets,
        val_data_dict=val_data_dict,
        max_new_tokens=cfg.training.max_new_tokens,
    )
    trainer.add_callback(eval_callback)

    # Initial evaluation (optional)
    if cfg.trainer_args.get("eval_on_start", False):
        print(f"\n{'='*80}")
        print("Initial Evaluation (Before Training)")
        print(f"{'='*80}")
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

    save_path = os.path.join(output_dir, "skill_tokens")
    adapter.save_skills(save_path)
    print(f"Saved skill tokens to: {save_path}")

    if wandb_run is not None:
        wandb_run.summary["skill_token_path"] = save_path
        wandb_run.finish()
        print(f"WandB run finished: {wandb_run.url}")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
