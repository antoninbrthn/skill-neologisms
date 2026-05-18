"""Evaluate zero-shot composition of independently learned skill neologisms vs ICL."""

import argparse
import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List

import matplotlib.pyplot as plt

from sequence_map_experiment.config import RESULTS_DIR, OUTPUT_DIR
from sequence_map_experiment.data import PRETRAIN_OPS, generate_sample_data_skill
from sequence_map_experiment.test_utils import generate_test_dataset
from sequence_map_experiment.model import load_any_model
from sequence_map_experiment.utils import (
    evaluate_model_kpass,
    load_cfg_from_dict,
)
from src.wandb_utils import load_wandb_data
from src.models.skill_token_model import load_skill_model_from_cfg

# Global experiment config
TAGS = ["skill-tokens"]
SKILL_NAMES = ["[SHIFT_RIGHT]", "[INVERT_POLARITY]"]

SEQ_LEN_RANGE = (2, 8)
MIN_OPS = 2
MAX_OPS = 2
N_TEST_SAMPLES = 50

MAX_K = 1  # pass@1

ICL_EXAMPLES_PER_SKILL = [10, 20, 50, 100]
ICL_EXAMPLES_POOL_SIZE = 10_000

RANDOM_SEED = 123
RESULTS_CSV = os.path.join(RESULTS_DIR, "skill_vs_icl_results_full.csv")
TEST_OPS = PRETRAIN_OPS


# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def expand_dataset_with_adapter(adapter, raw_dataset):
    return {k: adapter.expand_dataset(v) for k, v in raw_dataset.items()}


def build_icl_dataset(
    raw_test_dataset: Dict[int, List[str]],
    examples_dataset_skills: List[List[str]],
    n_examples_per_skill: int,
) -> Dict[int, List[str]]:
    """
    Create ICL prompts for a fixed number of examples per skill.
    """
    icl_dataset = {}

    for seq_len, test_samples in raw_test_dataset.items():
        prompts = []
        for test_sample in test_samples:
            sampled_examples = []
            for skill_examples in examples_dataset_skills:
                sampled_examples.extend(random.sample(skill_examples, n_examples_per_skill))
            random.shuffle(sampled_examples)
            prompt = "\n".join(sampled_examples + [test_sample])
            prompts.append(prompt)

        icl_dataset[seq_len] = prompts

    return icl_dataset


# Main experiment
def run_experiment(
    tags: List[str] = None,
    skill_names: List[str] = None,
    test_ops: List[str] = None,
    icl_examples_per_skill: List[int] = None,
    results_csv: str = None,
):
    tags = TAGS if tags is None else tags
    skill_names = SKILL_NAMES if skill_names is None else skill_names
    test_ops = TEST_OPS if test_ops is None else test_ops
    icl_examples_per_skill = ICL_EXAMPLES_PER_SKILL if icl_examples_per_skill is None else icl_examples_per_skill
    results_csv = RESULTS_CSV if results_csv is None else results_csv

    if len(skill_names) != 2:
        raise ValueError("Expected exactly two skill names for this composition experiment.")

    set_seed(RANDOM_SEED)
    all_results = []

    # Load configs and identify skill runs
    df_configs, _ = load_wandb_data(tags=tags, reload_cache=True)
    skill_configs = df_configs[df_configs["tags"].str.contains("skill")]

    # Initialize skill adapter once
    cfg = load_cfg_from_dict(skill_configs.iloc[0].to_dict())
    adapter = load_skill_model_from_cfg(cfg)

    # Shared test dataset (raw + expanded)
    raw_test_dataset = generate_test_dataset(
        main_ops=[skill_names[0]],
        other_ops=[skill_names[1]],
        num_samples=N_TEST_SAMPLES,
        seq_len_range=SEQ_LEN_RANGE,
        min_ops=MIN_OPS,
        max_ops=MAX_OPS,
        reject_len=[],
        task_token_length=1,
        output_token="=",
    )

    # Skill neologism evaluation
    for test_op in test_ops:
        print(f"\nEvaluating skill neologism for test op: {test_op}")
        skill_configs["dataset.test_op"].unique()
        cfgs_for_op = skill_configs[skill_configs["dataset.test_op"] == test_op]
        if len(cfgs_for_op) == 0:
            print(f"No skill configs found for test op {test_op}, skipping.")
            continue

        for _, row in cfgs_for_op.iterrows():
            skill_ckpt = os.path.join(row["output_dir"], "skill_tokens")
            print(f"Loading skill checkpoint from {skill_ckpt}")
            adapter.load_skills(skill_ckpt, overwrite_existing=True)

        expanded_test_dataset = expand_dataset_with_adapter(adapter, raw_test_dataset)

        adapter.set_skill_unembed_to_zero(verbose=False)
        model, tokenizer = adapter.model, adapter.tokenizer

        for seq_len, test_data in expanded_test_dataset.items():
            passk_results = evaluate_model_kpass(model, tokenizer, test_data, verbose=False, k=MAX_K, batch_size=64)
            for k in passk_results["pass_at_k"].keys():
                print(f'{test_op} | Seq len {seq_len} | Pass@{k}: {passk_results["pass_at_k"][k]:.4f} over {len(test_data)} samples')
                all_results.append(
                    {
                        "method": "Neologism",
                        "test_op": test_op,
                        "n_examples_per_skill": 0,
                        "seq_len": seq_len,
                        "accuracy": passk_results["pass_at_k"][k],
                        "k": k,
                        "num_total": len(test_data),
                    }
                )

    # ICL baseline
    pretrained_path = cfg.pretrained.checkpoint
    # Use local checkpoint if checkpoint path exists, otherwise treat as HF repo id
    local_model_path = os.path.join(OUTPUT_DIR, pretrained_path)
    if os.path.exists(local_model_path):
        model_dir = local_model_path
    else:
        model_dir = pretrained_path
    icl_model, icl_tokenizer, _ = load_any_model(model_dir)

    examples_dataset_skills = []
    for skill in skill_names:
        skill_examples = generate_sample_data_skill(
            main_ops=[skill],
            other_ops=PRETRAIN_OPS,
            num_samples=ICL_EXAMPLES_POOL_SIZE,
            seq_len=SEQ_LEN_RANGE,
            min_ops=1,
            max_ops=3,
            reject_len=[5, 7, 9],
            reject_ops=[],
            task_token_length=1,
            output_token="=",
        )
        examples_dataset_skills.append(skill_examples)

    for n_ex in icl_examples_per_skill:
        icl_dataset = build_icl_dataset(
            raw_test_dataset,
            examples_dataset_skills,
            n_examples_per_skill=n_ex,
        )

        for seq_len, prompts in icl_dataset.items():
            passk_results = evaluate_model_kpass(icl_model, icl_tokenizer, prompts, verbose=False, k=MAX_K, batch_size=64)
            for k in passk_results["pass_at_k"].keys():
                all_results.append(
                    {
                        "method": "ICL",
                        "test_op": "None",
                        "n_examples_per_skill": n_ex,
                        "seq_len": seq_len,
                        "accuracy": passk_results["pass_at_k"][k],
                        "k": k,
                        "num_total": len(prompts),
                    }
                )

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"Saved results to {results_csv}")

    return df


# Plotting
def plot_results(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))

    for method in ["Neologism", "ICL"]:
        sub = df[df["method"] == method]
        grouped = sub.groupby("seq_len")["accuracy"].mean()
        plt.plot(grouped.index, grouped.values, marker="o", label=method)

    plt.xlabel("Sequence length")
    plt.ylabel("Accuracy")
    plt.title("Skill Neologism vs ICL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    import seaborn as sns

    sns.lineplot(
        data=df,
        x="seq_len",
        y="accuracy",
        hue="test_op",
        style="method",
        markers=True,
        dashes=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate skill-token composition vs ICL.")
    parser.add_argument("--tags", nargs="+", default=TAGS, help="WandB tags for skill-token runs.")
    parser.add_argument(
        "--skills",
        nargs="+",
        default=SKILL_NAMES,
        help="Two independently trained skill names.",
    )
    parser.add_argument(
        "--test-ops",
        nargs="+",
        default=TEST_OPS,
        help="Held-out pretrain operations for evaluation.",
    )
    parser.add_argument(
        "--icl-examples-per-skill",
        nargs="+",
        type=int,
        default=ICL_EXAMPLES_PER_SKILL,
        help="ICL example counts per skill.",
    )
    parser.add_argument("--results-csv", default=RESULTS_CSV, help="Output CSV path.")
    args = parser.parse_args()

    if len(args.skills) != 2:
        raise ValueError("Expected exactly two skills for this composition experiment.")

    results_df = run_experiment(
        tags=args.tags,
        skill_names=args.skills,
        test_ops=args.test_ops,
        icl_examples_per_skill=args.icl_examples_per_skill,
        results_csv=args.results_csv,
    )
    plot_results(results_df)
