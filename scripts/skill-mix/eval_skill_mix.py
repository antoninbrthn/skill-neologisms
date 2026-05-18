"""
Run SkillMix evaluation pipeline.

Steps:
1. Load config (yaml + CLI overrides via OmegaConf)
2. Generate eval dataset (skill/topic combinations -> prompts)
3. Load model and generate completions (batched HF generate)
4. Optionally grade completions via API model (GPT)
5. Save everything (records, config, raw outputs) under output dir
"""

import os
import json
from datetime import datetime

import hydra
from omegaconf import OmegaConf
import pandas as pd
import wandb

from src.config import PROJECT_ROOT
from src.models.api_models import OpenAIModel
from src.wandb_utils import init_wandb
from src.skill_mix.custom_utils.data import generate_eval_dataset, load_skills
from src.skill_mix.custom_utils.generate import (
    generate_batch_hf,
    extract_answer,
    generate_batch_gpt,
)
from src.models.loading import load_any_model
from src.skill_mix.custom_utils.grade import load_grader, grade_all
from src.models.skill_token_model import insert_skill_tokens


def make_output_dir(cfg):
    """Create timestamped output directory."""
    base = cfg.output.get("dir", None)
    if base is None:
        model_short = cfg.model.name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(PROJECT_ROOT, "exports", "skill_mix", f"{model_short}_{timestamp}")
    os.makedirs(base, exist_ok=True)
    return base


def run(cfg):
    output_dir = make_output_dir(cfg)
    print(f"Output dir: {output_dir}")

    # Initialize wandb
    wandb_run = init_wandb(cfg, run_formula="{cfg.model.name}_k{cfg.skill_mix.k}_{timestamp}")

    # save resolved config (includes CLI overrides)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # 1. Generate eval dataset
    print("Building eval dataset...")
    dataset = generate_eval_dataset(cfg)
    print(f"  {len(dataset)} items")
    print(f"Sample 3 items:")
    for item in dataset[:3]:
        print(f"  Skills: {item['skills']}, Topic: {item['topic']}")

    # 2. Load model
    print(f"Loading model: {cfg.model.name}...")
    model, tokenizer, device, adapter = load_any_model(cfg.model)

    # 3. Generate completions
    gen_cfg = cfg.generation
    prompts = [d["prompt"] for d in dataset]
    # Insert skill tokens into prompts
    for skill_cfg in cfg.model.get("skills", []):
        if skill_cfg.get("prepend", False):
            # For prompt-tuning baseline: prepend skill token to every prompt
            prefix_tag = f"<|{skill_cfg.name}|>"
            prompts = [prefix_tag + p for p in prompts]
            prompts = [adapter._expand_skill_tokens(p) for p in prompts]
        else:  # For skill neologisms
            replace_str = skill_cfg.get("replace_str", None)
            if replace_str is not None:
                print(f"Replacing '{replace_str}' -> skill token <|{skill_cfg.name}|> in prompts...")
                ignore_str = skill_cfg.get("ignore_str", None)
                prompts = insert_skill_tokens(prompts, skill_cfg.name, replace_str, adapter, ignore_str=ignore_str)
            else:
                raise ValueError(f"WARNING: No 'replace_str' specified for skill '{skill_cfg.name}', skipping token insertion.")
    print(f"Generating completions (batch_size={gen_cfg.get('batch_size', 8)})...")
    if type(model) is OpenAIModel:
        model.max_tokens = gen_cfg.get("max_new_tokens", 512)
        completions = generate_batch_gpt(model, prompts, batch_size=gen_cfg.get("batch_size", 8))
    else:
        completions = generate_batch_hf(
            model,
            tokenizer,
            prompts,
            device,
            adapter=adapter,
            **gen_cfg,
        )

    # build records
    records = []
    for item, completion in zip(dataset, completions):
        rec = {
            "skills": item["skills"],
            "topic": item["topic"],
            "prompt": item["prompt"],
            "raw_completion": completion,
            "completion": extract_answer(completion),
            "k": len(item["skills"]),
        }
        records.append(rec)

    # save generation results
    gen_path = os.path.join(output_dir, "generations.json")
    with open(gen_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} generations to {gen_path}")

    # also save as CSV
    csv_records = [{**r, "skills": ", ".join(r["skills"])} for r in records]
    csv_path = os.path.join(output_dir, "generations.csv")
    pd.DataFrame(csv_records).to_csv(csv_path, index=False)

    # Log to wandb
    if wandb_run is not None:
        wandb.log(
            {
                "num_generations": len(records),
                "output_dir": output_dir,
                "generations_json": gen_path,
                "generations_csv": csv_path,
            }
        )

    # 4. Grade (optional)
    grade_cfg = cfg.get("grading", {})
    if grade_cfg.get("enabled", False):
        bs = grade_cfg.get("batch_size", 32)
        max_tokens = grade_cfg.get("max_tokens", 1024)
        print(f"Grading with {grade_cfg.grader_model}, batch_size={bs}, max_tokens={max_tokens}...")
        skills_dict = load_skills(cfg.skill_mix.get("skills_csv", None))
        grader = load_grader(
            grade_cfg.grader_model,
            use_batch=grade_cfg.get("use_batch", True),
            max_tokens=max_tokens,
        )
        records = grade_all(
            grader,
            records,
            skills_dict,
            grade_cfg.get("prompt_version", "gpt"),
            batch_size=bs,
        )

        # save graded results
        graded_path = os.path.join(output_dir, "graded.json")
        with open(graded_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"Saved graded results to {graded_path}")
        # log total API from grader.total_price in log.txt
        with open(os.path.join(output_dir, "log.txt"), "w") as f:
            # number of graded items
            num_graded = len(records)
            f.write(f"Graded {num_graded} items with {grade_cfg.grader_model}\n")
            f.write(f"Total API cost for grading: ${grader.total_price:.4f}\n")
        print(f"Total API cost for grading: ${grader.total_price:.4f}")

        # summary stats
        scores = [r["normalized_score"] for r in records]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"Average normalized score: {avg:.3f} ({len(scores)} items)")

        # Log grading results to wandb
        if wandb_run is not None:
            wandb.log(
                {
                    "graded_json": graded_path,
                    "num_graded": num_graded,
                    "average_normalized_score": avg,
                    "grading_api_cost": grader.total_price,
                }
            )
            # Log score distribution by k
            k_scores = {}
            for r in records:
                k = r["k"]
                if k not in k_scores:
                    k_scores[k] = []
                k_scores[k].append(r["normalized_score"])

            for k, k_score_list in k_scores.items():
                avg_k = sum(k_score_list) / len(k_score_list) if k_score_list else 0
                wandb.log({f"avg_score_k{k}": avg_k, f"num_items_k{k}": len(k_score_list)})

    print(f"Done. All outputs in {output_dir}")

    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "configs/skill_mix/"),
    config_name="default.yaml",
)
def main(cfg):
    print("Loaded config:", cfg)
    run(cfg)


if __name__ == "__main__":
    main()
