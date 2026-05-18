"""
Evaluate trained PEFT baselines (LoRA / Prompt Tuning) on SkillMix.

Steps:
  1. Fetch baseline training runs from wandb (by tags)
  2. For each run, extract: model name, PEFT adapter path, skill, peft type
  3. Build an OmegaConf config from the base eval_baseline_peft.yaml + per-run overrides
  4. Call eval_skill_mix.run() directly for each (run × k × split) combination
"""

import os
import sys
import argparse
import copy

import wandb
from omegaconf import OmegaConf

from src.config import PROJECT_ROOT
from src.wandb_utils import load_runs

SKILL_NAME_TO_SKILLMIX = {
    "stat_syllogism": "statistical syllogism",
    "modus": "modus ponens",
}
ALL_TRAINED_SKILLS = list(SKILL_NAME_TO_SKILLMIX.values())
OOD_TEST_SKILLS = ["emotional self regulation", "metaphor"]


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def _fetch_existing_eval_tag_sets(project="skill-neologisms-skillmix"):
    """Return a set of frozensets — each frozenset is the tag-set of a *finished*
    eval_baseline_peft run already on wandb.  Used by ``--skip_existing``."""
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WandB entity is required. Set WANDB_ENTITY before running this script.")

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": "eval_baseline_peft", "state": "finished"},
    )
    tag_sets = set()
    for run in runs:
        if run.tags:
            tag_sets.add(frozenset(run.tags))
    print(f"  Found {len(tag_sets)} existing finished eval_baseline_peft tag-sets on wandb")
    return tag_sets


def fetch_baseline_runs(tags, project="skill-neologisms-skillmix"):
    """Fetch baseline training runs from wandb and extract relevant metadata.

    Returns list of dicts with keys:
        run_id, run_name, model_name, peft_type, skill_name, peft_adapter_path
    """
    print(f"Fetching wandb runs  project={project}  tags={tags}")
    runs_dict = load_runs(run_tag=tags, project_name=project)
    print(f"  Found {len(runs_dict)} runs")

    entries = []
    for key, run in runs_dict.items():
        if run.state != "finished":
            print(f"  Skipping non-finished run: {run.name} (state={run.state})")
            continue

        summary = run.summary._json_dict
        config = run.config

        # Infer peft type from config
        peft_type = config.get("peft", {}).get("type", None)

        # pt_skillmodel runs store their path under skill_token_path
        adapter_path = summary.get("peft_adapter_path", None)
        skill_token_path = summary.get("skill_token_path", None)

        if adapter_path is None and skill_token_path is None:
            print(f"  Skipping run without peft_adapter_path or skill_token_path: {run.name}")
            continue

        # If we only have skill_token_path, treat this as pt_skillmodel
        if adapter_path is None and skill_token_path is not None:
            adapter_path = skill_token_path
            if peft_type is None:
                peft_type = "pt_skillmodel"

        # Extract model name from config
        model_name = config.get("model", {}).get("name", None)

        # Infer skill from dataset config if available, else from dataset_path
        skill_name = config.get("dataset", {}).get("skill_name", None)
        if skill_name is None:
            # Fallback: parse from training.dataset_path
            ds_path = config.get("training", {}).get("dataset_path", "")
            if "stat" in ds_path:
                skill_name = "stat_syllogism"
            elif "modus" in ds_path:
                skill_name = "modus"

        if skill_name is None or model_name is None:
            print(f"  Skipping run with missing metadata: {run.name}")
            continue

        entries.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "model_name": model_name,
                "peft_type": peft_type,
                "skill_name": skill_name,
                "peft_adapter_path": adapter_path,
                "wandb_tags": list(run.tags) if run.tags else [],
            }
        )

    print(f"  {len(entries)} usable runs")
    return entries


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def load_base_eval_config(config_name="eval_baseline_peft.yaml"):
    """Load the base eval_baseline_peft.yaml as an OmegaConf dict."""
    path = os.path.join(PROJECT_ROOT, "configs", "skill_mix", config_name)
    return OmegaConf.load(path)


def build_eval_config(entry, k, split, base_cfg):
    """Build a complete eval config for one (run × k × split) combination.

    Args:
        entry: dict from fetch_baseline_runs
        k: number of skills per combination
        split: "id" or "ood"
        base_cfg: base OmegaConf from eval_baseline_peft.yaml

    Returns:
        Resolved OmegaConf config ready for eval_skill_mix.run()
    """
    cfg = copy.deepcopy(base_cfg)

    # Model
    cfg.model.name = entry["model_name"]

    if entry["peft_type"] == "pt_skillmodel":
        # SkillTokenModel-based prompt tuning: load as skill model with a prepend-only PREFIX skill
        cfg.model.type = "skill"
        cfg.model.skills = [
            {
                "name": "PREFIX",
                "checkpoint_path": entry["peft_adapter_path"],
                "prepend": True,
            }
        ]
    else:
        cfg.model.type = "peft"
        cfg.model.peft_adapter_path = entry["peft_adapter_path"]

    # Set k from argument
    cfg.skill_mix.k = k
    # Work out which skill to use based on config
    if cfg.skill_mix.get("new_skills") is None:
        skill_mix_name = SKILL_NAME_TO_SKILLMIX[entry["skill_name"]]
        cfg.skill_mix.new_skills = [skill_mix_name]
        # Exclude all trained skills from the pool of "other" skills
        cfg.skill_mix.exclude_skills = ALL_TRAINED_SKILLS
    else:
        print("Using new_skills from config, ignoring skill_name from wandb run")
        print(f"New skills: {cfg.skill_mix.new_skills}")
        assert cfg.skill_mix.get("exclude_skills") is not None, "If using new_skills from config, must also specify exclude_skills"

    if split == "id":
        cfg.skill_mix.topics_txt = "src/skill_mix/topics_train.txt"
        # ID: use test_skills to hold out OOD skills (so only ID skills fill remaining slots)
        cfg.skill_mix.test_skills = OOD_TEST_SKILLS
    elif split == "ood":
        cfg.skill_mix.topics_txt = "src/skill_mix/topics_test.txt"
        # OOD: use train_skills to restrict to OOD skills only
        cfg.skill_mix.train_skills = OOD_TEST_SKILLS
    else:
        raise ValueError(f"Unknown split: {split}")

    # Wandb tags
    model_short = entry["model_name"].split("/")[-1]
    cfg.wandb.get("tags", []).extend(
        [
            f"peft_{entry['peft_type']}",
            f"skill_{entry['skill_name']}",
            f"k{k}",
            split,
            model_short,
        ]
    )
    cfg.wandb.notes = (
        f"Eval {entry['peft_type']} baseline | {entry['skill_name']} | "
        f"k={k} | {split} | model={entry['model_name']} | "
        f"train_run={entry['run_name']}"
    )

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT baselines on SkillMix")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["baseline"],
        help="Wandb tags to filter training runs (default: ['baseline'])",
    )
    parser.add_argument("--project", default="skill-neologisms-skillmix", help="Wandb project name")
    parser.add_argument(
        "--peft_type",
        choices=["lora", "prompt_tuning", "pt_skillmodel"],
        default=None,
        help="Filter to a specific PEFT type (default: all)",
    )
    parser.add_argument(
        "--skill",
        default=None,
        help="Filter to a specific skill name (stat_syllogism, modus, complex)",
    )
    parser.add_argument("--model", default=None, help="Filter to a specific model (substring match)")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[2],
        help="Value of k to evaluate (default: 2)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["id", "ood"],
        help="Eval splits (default: id ood)",
    )
    parser.add_argument(
        "--config_name",
        default="eval_baseline_peft.yaml",
        help="Name of the evaluation config file to use",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would run without actually running",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip jobs whose exact wandb tags already match a finished run",
    )
    parser.add_argument("--no_grading", action="store_true", help="Disable grading (generation only)")
    args = parser.parse_args()

    # Fetch runs
    entries = fetch_baseline_runs(args.tags, project=args.project)

    # Filter out PEFT-based Prompt Tuning runs (running PT via SkillTokenModel instead)
    entries = [e for e in entries if e["peft_type"] != "prompt_tuning"]

    # Apply filters
    if args.peft_type:
        entries = [e for e in entries if e["peft_type"] == args.peft_type]
    if args.skill:
        entries = [e for e in entries if e["skill_name"] == args.skill]
    if args.model:
        entries = [e for e in entries if args.model in e["model_name"]]

    if not entries:
        print("No matching runs found. Check tags/filters.")
        sys.exit(1)

    # Build eval jobs
    base_cfg = load_base_eval_config(config_name=args.config_name)
    if args.no_grading:
        base_cfg.grading.enabled = False

    eval_jobs = []
    for entry in entries:
        for k in args.k:
            for split in args.splits:
                cfg = build_eval_config(entry, k, split, base_cfg)
                eval_jobs.append((entry, k, split, cfg))

    # Optionally skip jobs that already have a matching finished run on wandb
    if args.skip_existing:
        existing_tag_sets = _fetch_existing_eval_tag_sets(args.project)
        before = len(eval_jobs)
        eval_jobs = [(entry, k, split, cfg) for entry, k, split, cfg in eval_jobs if frozenset(cfg.wandb.tags) not in existing_tag_sets]
        skipped = before - len(eval_jobs)
        if skipped:
            print(f"  --skip_existing: skipping {skipped} already-finished jobs")

    print(f"\n{'=' * 80}")
    print(f"Evaluation plan: {len(eval_jobs)} jobs")
    print(f"  Runs: {len(entries)}")
    print(f"  k values: {args.k}")
    print(f"  Splits: {args.splits}")
    print(f"{'=' * 80}")

    for i, (entry, k, split, cfg) in enumerate(eval_jobs):
        model_short = entry["model_name"].split("/")[-1]
        print(f"  [{i+1}/{len(eval_jobs)}] {entry['peft_type']:15s} | {entry['skill_name']:15s} | " f"k={k} | {split:3s} | {model_short}")

    if args.dry_run:
        print("\n[DRY RUN] — no evaluations launched.")
        # Print one example config
        if eval_jobs:
            print("\nExample resolved config:")
            print(OmegaConf.to_yaml(eval_jobs[0][3]))
        return

    # Run evaluations
    # Import run() from eval_skill_mix (deferred to avoid heavy imports during dry run).
    # scripts/skill-mix/ has a hyphen so we use importlib.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_skill_mix",
        os.path.join(PROJECT_ROOT, "scripts", "skill-mix", "eval_skill_mix.py"),
    )
    rsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rsm)
    eval_skill_mix_eval = rsm.run

    for i, (entry, k, split, cfg) in enumerate(eval_jobs):
        # Ensure any lingering wandb run from a prior job (or crash) is closed
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass

        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{len(eval_jobs)}] Running eval: " f"{entry['peft_type']} | {entry['skill_name']} | k={k} | {split}")
        print(f"{'=' * 80}")
        try:
            eval_skill_mix_eval(cfg)
        except Exception as e:
            print(f"\n⚠️  Eval failed: {e}")
            import traceback

            traceback.print_exc()
            # Make sure we close the wandb run so the next job starts clean
            try:
                wandb.finish()
            except Exception:
                pass
            continue

    print(f"\n{'=' * 80}")
    print(f"All evaluations complete ({len(eval_jobs)} jobs)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
