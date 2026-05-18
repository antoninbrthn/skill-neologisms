"""
Create a single csv dataset by concatenating generated data from wandb runs under a given tag.
"""

import os
import pandas as pd
import wandb
import argparse

from src.config import PROJECT_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset from wandb runs")
    parser.add_argument("--tag", type=str, default="train_data_gpt5", help="Tag to filter wandb runs by")
    args = parser.parse_args()

    tag = args.tag
    project_name = "skill-neologisms-skillmix"
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WandB entity is required. Set WANDB_ENTITY before running this script.")
    output_dir = os.path.join(PROJECT_ROOT, "exports", "skill_mix", f"train_data_gpt5")

    # 1. Load wandb runs
    api = wandb.Api()
    # Filter runs by tags
    if isinstance(tag, str):
        runs = api.runs(f"{entity}/{project_name}", filters={"tags": tag})
    elif isinstance(tag, list):
        runs = api.runs(f"{entity}/{project_name}", filters={"tags": {"$in": tag}})
    else:
        raise ValueError("tag must be a string or list of strings")
    print(f"Found {len(runs)} runs with tag '{tag}' in {entity}/{project_name}")
    all_generations = []
    for run in runs:
        if run.state != "finished":
            continue
        summary = run.summary["generations_csv"]
        # load
        generations = pd.read_csv(summary)
        all_generations.append(generations)

    if len(all_generations) == 0:
        raise ValueError(f"No runs found for tag '{tag}' in {entity}/{project_name}.")

    all_generations_df = pd.concat(all_generations, ignore_index=True)
    print(f"Loaded {len(all_generations_df)} generations from {len(runs)} runs")

    # export to csv
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{tag}.csv")
    all_generations_df.to_csv(output_path, index=False)
    print(f"Exported dataset to {output_path}")
