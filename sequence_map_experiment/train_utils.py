"""Shared helpers for training scripts."""

from typing import List, Optional

import pandas as pd
import torch
import wandb
from transformers import TrainerCallback

from sequence_map_experiment.utils import evaluate_model_batch
from sequence_map_experiment.test_utils import test_on_skill_1op_varying_lengths


def expand_skill_token_texts(texts: List[str], skill_name: str, adapter) -> List[str]:
    """Replace a skill marker with the special skill token and expand it."""
    expanded = [text.replace(skill_name, f"<|{skill_name}|>") for text in texts]
    return [adapter._expand_skill_tokens(text) for text in expanded]


def prepend_prefix_texts(texts: List[str], prefix_name: str, adapter) -> List[str]:
    """Prepend a prefix skill token to every text and expand it."""
    prefix_tag = f"<|{prefix_name}|>"
    prefixed = [prefix_tag + text for text in texts]
    return [adapter._expand_skill_tokens(text) for text in prefixed]


class KOpEvalCallback(TrainerCallback):
    """Callback to evaluate validation/test k-op sets at the end of each epoch."""

    def __init__(
        self,
        adapter,
        cfg,
        skill_op,
        test_op,
        test_datasets,
        val_data_dict,
        max_new_tokens=10,
        prefix_name: Optional[str] = None,
        prompt_format_type: Optional[str] = None,
    ):
        self.adapter = adapter
        self.cfg = cfg
        self.skill_op = skill_op
        self.test_op = test_op
        self.test_datasets = test_datasets
        self.val_data_dict = val_data_dict
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_ops = cfg.dataset.get("max_ops", 2)
        self.prefix_name = prefix_name
        self.prompt_format_type = prompt_format_type

    def run_evaluation(self, epoch):
        print(f"\n{'=' * 80}")
        print(f"End of Epoch {epoch} Evaluation")
        print(f"{'=' * 80}")

        print(f"\n[Epoch {epoch}] Evaluating on validation sets...")
        for num_ops in self.val_data_dict.keys():
            val_data = self.val_data_dict[num_ops]
            val_acc, _ = evaluate_model_batch(
                self.adapter.model,
                self.adapter.tokenizer,
                val_data,
                device=self.device,
                verbose=False,
            )
            print(f"  Val {num_ops}-op accuracy: {val_acc:.4f}")
            if wandb.run is not None:
                wandb.log({"epoch": epoch, f"val_{num_ops}op/accuracy": val_acc})

        print("\n" + "=" * 80)
        print("Testing on 1-op tasks with varying sequence lengths...")
        print("=" * 80)

        results_df = test_on_skill_1op_varying_lengths(
            self.adapter,
            self.cfg,
            num_samples=50,
            device=self.device,
            verbose=False,
            prefix_name=self.prefix_name,
            prompt_format_type=self.prompt_format_type,
        )

        print("\n" + "=" * 80)
        print("Accuracy by task and sequence length:")
        print("=" * 80)
        print(results_df.to_string(float_format=lambda x: f"{x:.2%}"))

        print("\nMean accuracy per sequence length:")
        len_means = results_df.groupby("seq_len")["accuracy"].mean()
        for seq_len, acc in len_means.items():
            print(f"  Length {seq_len}: {acc:.2%}")

        print(f"\nOverall mean accuracy (1-op tasks): {results_df['accuracy'].mean():.2%}")

        for seq_len, acc in len_means.items():
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "test_1op/mean_accuracy": acc,
                        "test_1op/seq_len": seq_len,
                        f"test_1op_detailed/mean_accuracy_len_{seq_len}": acc,
                    }
                )

        if wandb.run is not None:
            wandb.run.summary["test_1op/results_table"] = wandb.Table(dataframe=results_df)

        for num_ops in self.test_datasets.keys():
            print(f"\n[Epoch {epoch}] Evaluating on test_{num_ops}op...")
            results = []

            for seq_len, perm_dict in self.test_datasets[num_ops].items():
                for perm_key, test_data in perm_dict.items():
                    acc, _ = evaluate_model_batch(
                        self.adapter.model,
                        self.adapter.tokenizer,
                        test_data,
                        device=self.device,
                        verbose=False,
                    )
                    results.append(
                        {
                            "epoch": epoch,
                            "num_ops": num_ops,
                            "seq_len": seq_len,
                            "permutation": perm_key,
                            "accuracy": acc,
                        }
                    )
                    print(f"  Seq_len {seq_len}, ops [{perm_key}]: {acc:.4f}")

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "epoch": epoch,
                                f"test_{num_ops}op/seq_len": seq_len,
                                f"test_{num_ops}op/{perm_key}": acc,
                                f"test_{num_ops}op_detailed/{perm_key}_len_{seq_len}": acc,
                            }
                        )

            dt_results = pd.DataFrame(results)
            if wandb.run is not None:
                wandb.run.summary[f"test_{num_ops}op/results_table"] = wandb.Table(dataframe=dt_results)

            if len(results) > 0:
                mean_acc = sum([r["accuracy"] for r in results]) / len(results)
                print(f"\n[Epoch {epoch}] Mean test_{num_ops}op accuracy: {mean_acc:.4f}")
                if wandb.run is not None:
                    wandb.log({"epoch": epoch, f"test_{num_ops}op/mean_accuracy": mean_acc})

        print(f"{'=' * 80}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        self.run_evaluation(epoch)
