"""
Train prompt tuning baseline using SkillTokenModel on k-operation tasks.

We ran into issues when using PEFT's PromptTuningConfig implementation, so instead
we train the prompt tuning baseline using a SkillTokenModel backend:
- create a learnable PREFIX skill with SkillTokenModel,
- prepend it to every prompt and keep the rest of the prompt as is.

Usage:
    PYTHONPATH=. python sequence_map_experiment/train_prompt_tuning.py --config_name baseline_prompt_tuning.yaml
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import TrainerCallback
from trl import SFTConfig

from sequence_map_experiment.config import OUTPUT_DIR
from sequence_map_experiment.data import (
    generate_test_datasets,
    generate_train_val_datasets,
    split_on_output,
)
from sequence_map_experiment.train_utils import KOpEvalCallback, prepend_prefix_texts
from sequence_map_experiment.model import load_any_model
from src.config import PROJECT_ROOT
from src.models.skill_token_model import SkillTokenModel
from src.trainer_utils import SkillTokenTrainer
from src.wandb_utils import init_wandb


class SkillSaveCallback(TrainerCallback):
    """Save prefix skill embeddings during training."""

    def __init__(self, adapter: SkillTokenModel, save_strategy: str = "no", save_steps: int = 500):
        self.adapter = adapter
        self.save_strategy = save_strategy
        self.save_steps = save_steps

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.save_strategy == "epoch":
            epoch = int(state.epoch)
            save_path = os.path.join(args.output_dir, f"skill_tokens_epoch-{epoch}")
            self.adapter.save_skills(save_path)
            print(f"\n[Epoch {epoch}] Saved skill tokens to: {save_path}\n")

    def on_step_end(self, args, state, control, **kwargs):
        if self.save_strategy == "steps" and state.global_step % self.save_steps == 0 and state.global_step > 0:
            save_path = os.path.join(args.output_dir, f"skill_tokens_step-{state.global_step}")
            self.adapter.save_skills(save_path)
            print(f"\n[Step {state.global_step}] Saved skill tokens to: {save_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train prompt tuning baseline using SkillTokenModel on k-operation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config file name (e.g., baseline_prompt_tuning.yaml)",
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

    config_dir = os.path.join(PROJECT_ROOT, "sequence_map_experiment", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    print(f"\nConfiguration loaded.")
    print(f"\n{OmegaConf.to_yaml(cfg)}")

    skill_op = cfg.dataset.skill_op
    test_op = cfg.dataset.test_op
    prefix_name = cfg.skill.name

    if skill_op is None:
        raise ValueError("dataset.skill_op must be specified in the config")
    if test_op is None:
        raise ValueError("dataset.test_op must be specified in the config")

    print(f"\nKey parameters:")
    print(f"  Skill op: {skill_op}")
    print(f"  Test op: {test_op}")
    print(f"  Prefix skill: {prefix_name}")

    model_name = cfg.model.name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading base model: {model_name} on {device}...")
    model, tokenizer, model_name = load_any_model(cfg.pretrained.checkpoint, cfg)

    adapter = SkillTokenModel(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
    )
    adapter.model_cfg = OmegaConf.to_container(cfg, resolve=True)

    prefix_length = cfg.skill.length
    init_method = cfg.skill.get("init_method", "rand")
    print(f"Creating prefix skill '{prefix_name}' with length={prefix_length}, init={init_method}")
    adapter.create_skill(
        skill_id=prefix_name,
        length=prefix_length,
        desc=cfg.skill.get("description", "prompt_tuning_prefix"),
        init_method=init_method,
    )

    resume_from_checkpoint = cfg.skill.get("checkpoint_path", None)
    if resume_from_checkpoint is not None:
        print(f"\n>>> Resuming from checkpoint: {resume_from_checkpoint}")
        skill_subdir = os.path.join(resume_from_checkpoint, "skill_tokens")
        if os.path.isdir(skill_subdir):
            print(f"  Restoring skill token embeddings from: {skill_subdir}")
            adapter.load_skills(skill_subdir, overwrite_existing=True)
        else:
            print("  WARNING: No skill_tokens/ subdirectory in checkpoint; skill embeddings NOT restored.")
    else:
        print("\nNo checkpoint_path provided - training from scratch.")

    num_train_samples = cfg.dataset.get("num_samples", 100_000)
    num_val_samples = cfg.dataset.get("val_dataset_size", 1000)
    num_test_samples = cfg.dataset.get("test_dataset_size", 200)
    task_token_length = adapter.model_cfg.get("adaptation", {}).get("task_length", 1)
    print(f"{task_token_length=}")

    train_data_dict, val_data_dict = generate_train_val_datasets(
        cfg,
        skill_op,
        test_op,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        task_token_length=task_token_length,
    )

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

    test_datasets = generate_test_datasets(
        cfg,
        skill_op,
        test_op,
        num_samples=num_test_samples,
        task_token_length=task_token_length,
    )

    train_data = prepend_prefix_texts(train_data, prefix_name, adapter)
    val_data = prepend_prefix_texts(val_data, prefix_name, adapter)

    for num_ops in val_data_dict.keys():
        val_data_dict[num_ops] = prepend_prefix_texts(val_data_dict[num_ops], prefix_name, adapter)

    for num_ops in test_datasets.keys():
        for seq_len in test_datasets[num_ops]:
            for perm_key in test_datasets[num_ops][seq_len]:
                test_datasets[num_ops][seq_len][perm_key] = prepend_prefix_texts(test_datasets[num_ops][seq_len][perm_key], prefix_name, adapter)

    train_prompts, train_labels = split_on_output(train_data)
    val_prompts, val_labels = split_on_output(val_data)

    train_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(train_prompts, train_labels)])
    val_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(val_prompts, val_labels)])

    print("\nRandom 5 training samples:")
    print("  " + "\n  ".join(np.random.choice(train_data, size=5, replace=False)))
    print("\nRandom 5 validation samples:")
    print("  " + "\n  ".join(np.random.choice(val_data, size=5, replace=False)))

    wandb_run = None
    if cfg.wandb.get("enabled", False):
        model_name_str = model_name.split("/")[-1].replace(".", "_")
        wandb_run = init_wandb(cfg, run_formula=f"pt_{model_name_str}_l{prefix_length}_{{timestamp}}")

    trainer_args_dict = OmegaConf.to_container(cfg.trainer_args, resolve=True)

    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_suffix = f"_{wandb_run.name.replace('/', '-')}" if wandb_run is not None else ""
    output_dir = os.path.join(OUTPUT_DIR, cfg.output.save_dir, f"{ts_str}{run_suffix}")

    trainer_args_dict.update(
        {
            "output_dir": output_dir,
            "report_to": "wandb" if wandb_run is not None else None,
            "completion_only_loss": True,
            "save_strategy": trainer_args_dict.get("save_strategy", "no"),
        }
    )

    training_args = SFTConfig(**trainer_args_dict)

    print(f"\nTraining arguments configured:")
    print(f"  Prefix length: {prefix_length}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output dir: {training_args.output_dir}")

    trainer = SkillTokenTrainer(
        adapter=adapter,
        model=adapter.model,
        processing_class=adapter.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    skill_save_strategy = cfg.skill.get("save_strategy", "no")
    skill_save_steps = cfg.skill.get("save_steps", 500)
    save_callback = SkillSaveCallback(adapter, save_strategy=skill_save_strategy, save_steps=skill_save_steps)
    trainer.add_callback(save_callback)

    eval_callback = KOpEvalCallback(
        adapter=adapter,
        cfg=cfg,
        prefix_name=prefix_name,
        skill_op=skill_op,
        test_op=test_op,
        test_datasets=test_datasets,
        val_data_dict=val_data_dict,
        max_new_tokens=cfg.training.max_new_tokens,
    )
    trainer.add_callback(eval_callback)

    if cfg.trainer_args.get("eval_on_start", False):
        print(f"\n{'=' * 80}")
        print("Initial Evaluation (Before Training)")
        print(f"{'=' * 80}")
        eval_callback.run_evaluation(epoch=0)

    print(f"\n{'=' * 80}")
    print("Training with HuggingFace Trainer...")
    print(f"{'=' * 80}\n")

    train_result = trainer.train()

    print(f"\n{'=' * 80}")
    print("Training completed!")
    print(f"{'=' * 80}")
    print(f"Train metrics: {train_result.metrics}")

    print(f"\n{'=' * 80}")
    print("Saving results...")
    print(f"{'=' * 80}")

    save_path = os.path.join(output_dir, "skill_tokens")
    adapter.save_skills(save_path)
    print(f"Saved prefix skill tokens to: {save_path}")

    if wandb_run is not None:
        wandb_run.summary["skill_token_path"] = save_path
        wandb_run.finish()
        print(f"WandB run finished: {wandb_run.url}")

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
