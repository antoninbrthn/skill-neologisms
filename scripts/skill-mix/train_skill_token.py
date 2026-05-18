"""Train skill neologisms from a CSV dataset (Skill-mix)."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from omegaconf import OmegaConf
import hydra
from transformers import TrainerCallback
from trl import SFTConfig

from src.config import PROJECT_ROOT
from src.models.skill_token_model import SkillTokenModel, insert_skill_tokens
from src.trainer_utils import SkillTokenTrainer
from src.wandb_utils import init_wandb


class SkillSaveCallback(TrainerCallback):
    """Save skill token embeddings during training."""

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


def load_csv_dataset(path: str, col_prompt: str, col_completion: str):
    """Load a csv and return lists of (prompt, completion) strings."""
    df = pd.read_csv(path)
    assert col_prompt in df.columns, f"Column '{col_prompt}' not found in {path}. Columns: {list(df.columns)}"
    assert col_completion in df.columns, f"Column '{col_completion}' not found in {path}. Columns: {list(df.columns)}"
    # Drop rows where either column is NaN
    n_before = len(df)
    df = df.dropna(subset=[col_prompt, col_completion])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with NaN in '{col_prompt}' or '{col_completion}'")
    prompts = df[col_prompt].astype(str).tolist()
    completions = df[col_completion].astype(str).tolist()
    return prompts, completions


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "configs/skill_mix/"),
    config_name="default.yaml",
)
def main(cfg):
    print("=" * 80)
    print("Skill Token Training (skill-mix)")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    # Build SkillTokenModel
    model_name = cfg.model.name
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"\nLoading base model: {model_name} on {device}...")
    adapter = SkillTokenModel(model_name=model_name, device=device)

    # Register / load each skill listed in cfg.model.skills
    for skill_cfg in cfg.model.skills:
        cp = skill_cfg.get("checkpoint_path", None)
        if cp is not None:
            print(f"Loading skill '{skill_cfg.name}' from checkpoint: {cp}")
            adapter.load_skills(cp, overwrite_existing=True)
        else:
            length = skill_cfg.get("length", 10)
            print(f"Creating new skill '{skill_cfg.name}' with length={length}")
            adapter.create_skill(skill_id=skill_cfg.name, length=length, desc=skill_cfg.name)

    print(f"Registered skills: {list(adapter.skill_tokens.keys())}")

    # Load CSV dataset
    dataset_path = cfg.training.dataset_path
    col_prompt = cfg.training.get("col_prompt", "prompt")
    col_completion = cfg.training.get("col_completion", "raw_completion")

    print(f"\nLoading dataset from: {dataset_path}")
    prompts, completions = load_csv_dataset(dataset_path, col_prompt, col_completion)
    print(f"  Loaded {len(prompts)} samples")

    # Insert skill tokens into prompts
    for skill_cfg in cfg.model.skills:
        replace_str = skill_cfg.get("replace_str", None)
        if replace_str is not None:
            print(f"Replacing '{replace_str}' -> skill token <|{skill_cfg.name}|> in prompts...")
            old_prompts = prompts
            prompts = insert_skill_tokens(prompts, skill_cfg.name, replace_str, adapter)
            # assert that all prompts were modified (i.e. replace_str was found in each prompt)
            modified = [p != op for p, op in zip(prompts, old_prompts)]
            n_modified = sum(modified)
            if n_modified == 0:
                raise ValueError(
                    f"ERROR: No prompts were modified when inserting skill token for '{skill_cfg.name}'. Check that 'replace_str'='{replace_str}' is correct and appears in the prompts."
                )
            elif n_modified < len(prompts):
                print(
                    f"  WARNING: Only {n_modified}/{len(prompts)} prompts were modified when inserting skill token for '{skill_cfg.name}'. Check that 'replace_str'='{replace_str}' is correct and appears in the prompts."
                )
                print("Filtering prompts to only those containing the replace_str for training...")
                prompts = [p for i, p in enumerate(prompts) if modified[i]]
                completions = [c for i, c in enumerate(completions) if modified[i]]
                print(f"  {len(prompts)} prompts remaining after filtering.")
        else:
            print(f"WARNING: No 'replace_str' specified for skill '{skill_cfg.name}', skipping token insertion.")

    # Train / val split
    val_frac = cfg.training.get("val_fraction", 0.1)
    n_val = max(1, int(len(prompts) * val_frac))
    indices = np.random.RandomState(123).permutation(len(prompts))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_prompts = [prompts[i] for i in train_idx]
    train_completions = [completions[i] for i in train_idx]
    val_prompts = [prompts[i] for i in val_idx]
    val_completions = [completions[i] for i in val_idx]

    print(f"\n  Train: {len(train_prompts)} samples")
    print(f"  Val:   {len(val_prompts)} samples")

    # Build HuggingFace Datasets
    train_dataset = Dataset.from_list([{"prompt": p, "completion": c} for p, c in zip(train_prompts, train_completions)])
    val_dataset = Dataset.from_list([{"prompt": p, "completion": c} for p, c in zip(val_prompts, val_completions)])
    # shuffle
    train_dataset = train_dataset.shuffle(seed=123)
    val_dataset = val_dataset.shuffle(seed=123)

    # Print a few samples
    print("\nSample training prompts (first 3):")
    for i in range(min(3, len(train_prompts))):
        preview = train_prompts[i][:500].replace("\n", "\\n")
        print(f"  [{i}] {preview}...")

    # WandB
    wandb_run = None
    if cfg.wandb.get("enabled", False):
        model_name_str = model_name.split("/")[-1].replace(".", "_")
        skill_names = "_".join([s.name for s in cfg.model.get("skills", [])])
        wandb_run = init_wandb(cfg, run_formula=f"smix_train_{model_name_str}_{skill_names}_{{timestamp}}")

    # Training arguments
    trainer_args_dict = OmegaConf.to_container(cfg.trainer_args, resolve=True)

    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_output_dir = os.path.join(PROJECT_ROOT, "exports", "skill_mix", f"train_{ts_str}")
    output_dir = cfg.output.get("dir", None) or default_output_dir
    if wandb_run is not None:
        output_dir = output_dir + f"_{wandb_run.name}"

    trainer_args_dict.update(
        {
            "output_dir": output_dir,
            "report_to": "wandb" if (wandb_run is not None) else None,
            "completion_only_loss": True,
            "save_strategy": trainer_args_dict.get("save_strategy", "no"),
        }
    )

    training_args = SFTConfig(**trainer_args_dict)

    print(f"\nTraining config:")
    print(f"  Epochs:        {training_args.num_train_epochs}")
    print(f"  Batch size:    {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output dir:    {training_args.output_dir}")

    # Create trainer
    trainer = SkillTokenTrainer(
        adapter=adapter,
        model=adapter.model,
        processing_class=adapter.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Add skill-save callback
    skill_save_strategy = cfg.training.get("skill_save_strategy", "epoch")
    skill_save_steps = cfg.training.get("skill_save_steps", 500)
    save_callback = SkillSaveCallback(adapter, save_strategy=skill_save_strategy, save_steps=skill_save_steps)
    trainer.add_callback(save_callback)

    # Train
    print(f"\n{'=' * 80}")
    print("Starting training...")
    print(f"{'=' * 80}\n")

    train_result = trainer.train()

    print(f"\n{'=' * 80}")
    print("Training completed!")
    print(f"{'=' * 80}")
    print(f"Train metrics: {train_result.metrics}")

    # Save final skill tokens
    save_path = os.path.join(output_dir, "skill_tokens")
    adapter.save_skills(save_path)
    print(f"\nSaved final skill tokens to: {save_path}")

    if wandb_run is not None:
        wandb_run.summary["skill_token_path"] = save_path
        wandb_run.finish()
        print(f"WandB run finished: {wandb_run.url}")

    print(f"\n{'=' * 80}")
    print("Done!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
