"""Aggregate and plot neologisms vs baselines on Skill-mix from wandb runs."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from src.plot_utils import format_fig, COLS
from src.config import PROJECT_ROOT

COMBINE_TWO_TAGS = ["eval_trained_combine_two", "eval_baseline_combine_two"]
TARGET_NEW_SKILLS = "statistical syllogism, modus ponens"

SLUG_TO_SKILL = {  # normalize skill names
    "stat_syllogism": "statistical syllogism",
    "modus": "modus ponens",
    "modus_ponens": "modus ponens",
}

PEFT_LABEL = {
    "lora": "LoRA",
    "pt_skillmodel": "PromptTuning",  # PT via SkillTokenModel adapter
}


def _normalize_skill_name(skill: str) -> str:
    if skill in SLUG_TO_SKILL:
        return SLUG_TO_SKILL[skill]
    if skill.lower() in SLUG_TO_SKILL:
        return SLUG_TO_SKILL[skill.lower()]
    return skill.replace("_", " ")


def _normalize_criteria_key(key: str) -> str:
    return key.strip().lstrip("-").strip()


def _find_skill_score(per_criteria: Dict, skill: str) -> Optional[int]:
    target = _normalize_criteria_key(f"correctly illustrates {skill}")
    for key, value in per_criteria.items():
        if _normalize_criteria_key(key) == target:
            return int(bool(value))
    return None


def _load_records_from_run(run, downloads_dir: str) -> Optional[List[Dict]]:
    summary = run.summary._json_dict
    graded_path = summary.get("graded_json")
    if graded_path and os.path.exists(graded_path):
        with open(graded_path, "r") as f:
            return json.load(f)

    try:
        os.makedirs(downloads_dir, exist_ok=True)
        downloaded = run.file("graded.json").download(root=downloads_dir, replace=True)
        with open(downloaded.name, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _score_records(records: List[Dict]) -> List[Dict]:
    per_skill = {}
    all_correct = 0
    all_total = 0

    for item in records:
        per_criteria = item.get("per_criteria") or {}
        sample_skills = item.get("skills") or []
        if not per_criteria or not sample_skills:
            continue

        all_true = True
        for skill in sample_skills:
            score = _find_skill_score(per_criteria, skill)
            if score is None:
                all_true = False
                continue
            if score == 0:
                all_true = False
            correct, total = per_skill.get(skill, (0, 0))
            per_skill[skill] = (correct + score, total + 1)

        all_total += 1
        all_correct += int(all_true)

    rows = []
    rows.append(
        {
            "metric": "all_skills",
            "accuracy": (all_correct / all_total) if all_total > 0 else 0.0,
            "n": all_total,
        }
    )
    for skill, (correct, total) in sorted(per_skill.items()):
        rows.append(
            {
                "metric": skill,
                "accuracy": (correct / total) if total > 0 else 0.0,
                "n": total,
            }
        )
    return rows


def _infer_baseline_train_skill(tags: Iterable[str]) -> str:
    for tag in tags:
        if tag.startswith("skill_"):
            return _normalize_skill_name(tag[len("skill_") :])
    return "unknown"


def _infer_peft_type(tags: Iterable[str], config: Dict) -> Optional[str]:
    for tag in tags:
        if tag.startswith("peft_"):
            return tag[len("peft_") :]
    return config.get("peft", {}).get("type")


def _infer_method_and_skill(run) -> Tuple[str, str]:
    tags = run.tags or []
    config = run.config

    peft_type = _infer_peft_type(tags, config)
    if peft_type is not None:
        train_skill = _infer_baseline_train_skill(tags)
        method = PEFT_LABEL.get(peft_type, peft_type)
        return method, train_skill

    model_cfg = config.get("model", {})
    if model_cfg.get("type") == "skill":
        skill_cfgs = model_cfg.get("skills", [])
        skill_names = sorted({_normalize_skill_name(s.get("name", "")) for s in skill_cfgs if s.get("name")})
        if len(skill_names) == 0:
            return "Skill Neologisms", "unknown"
        if len(skill_names) == 1:
            return "Skill Neologisms", skill_names[0]
        return "Skill Neologisms", " + ".join(skill_names)

    return "Baseline", "Unknown"


def _fetch_runs(entity: str, project: str, tags: List[str]):
    api = wandb.Api()
    return api.runs(
        f"{entity}/{project}",
        filters={"$and": [{"state": "finished"}, {"tags": {"$in": tags}}]},
    )


def collect_results(entity: str, project: str) -> pd.DataFrame:
    runs = _fetch_runs(entity=entity, project=project, tags=COMBINE_TWO_TAGS)
    downloads_dir = os.path.join(PROJECT_ROOT, "exports", "skill_mix", "wandb_downloads")
    rows = []

    for run in runs:
        records = _load_records_from_run(run, downloads_dir=downloads_dir)
        if records is None:
            print(f"Skipping {run.name} ({run.id}): graded results not found")
            continue

        method, trained_skill = _infer_method_and_skill(run)

        k_value = run.config.get("skill_mix", {}).get("k")
        new_skills = run.config.get("skill_mix", {}).get("new_skills", [])
        if new_skills and ", ".join(new_skills) != TARGET_NEW_SKILLS:
            continue
        metric_rows = _score_records(records)

        for metric_row in metric_rows:
            rows.append(
                {
                    "method": method,
                    "trained_skill": trained_skill,
                    "k": k_value,
                    "skill": metric_row["metric"],
                    "accuracy": metric_row["accuracy"],
                }
            )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze retained SkillMix rebuttal evaluations from WandB.")
    parser.add_argument("--project", default="skill-neologisms-skillmix")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "exports", "skill_mix", "analysis"),
    )
    return parser.parse_args()


def skillmix_combine_two(df_skills):
    TICK_FONTSIZE = 16
    AXIS_FONTSIZE = 18
    # Method display names
    METHOD_LABELS = {
        "Baseline": "Base\nModel",
        "LoRA": "LoRA",
        "PT (SkillModel)": "Prompt\nTuning",
        "Skill Neologisms": "Skill\nNeologisms",
    }

    # Pivot accuracy by evaluated skill
    pivot = df_skills.groupby(["method", "trained_skill", "skill"])["accuracy"].mean().unstack("skill").reset_index()

    # Determine trained-skill group for sorting
    def _group_order(trained_skill):
        if trained_skill in (None, "None", float("nan")):
            return 0
        ts = str(trained_skill)
        if "modus" in ts and "stat" in ts:
            return 3
        if "modus" in ts:
            return 1
        if "stat" in ts:
            return 2
        return 0

    pivot["_group"] = pivot["trained_skill"].apply(_group_order)

    # Within each group order by method for consistency
    method_order = list(METHOD_LABELS.keys())
    pivot["_method_order"] = pivot["method"].map({m: i for i, m in enumerate(method_order)})
    pivot = pivot.sort_values(["_group", "_method_order"]).reset_index(drop=True)

    # Build plot arrays
    labels = pivot["method"].map(METHOD_LABELS).fillna(pivot["method"]).tolist()
    mp_vals = pivot.get("modus ponens", pd.Series(0.0, index=pivot.index)).fillna(0).values
    ss_vals = pivot.get("statistical syllogism", pd.Series(0.0, index=pivot.index)).fillna(0).values
    al_vals = pivot.get("all_skills", pd.Series(0.0, index=pivot.index)).fillna(0).values

    n = len(pivot)
    x = np.arange(n)
    width = 0.25

    # Build group metadata dynamically from _group column
    group_label_map = {
        0: "No training",
        1: "Trained on\nModus Ponens",
        2: "Trained on\nStat. Syllogism",
        3: "Using both\nskill neologisms",
    }
    groups = []
    for g_id, g_label in sorted(group_label_map.items()):
        idxs = pivot.index[pivot["_group"] == g_id].tolist()
        if idxs:
            groups.append((idxs[0], idxs[-1], g_label))

    # Separator positions fall between consecutive groups
    separators = [(groups[i][1] + groups[i + 1][0]) / 2 for i in range(len(groups) - 1)]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))

    bar_specs = [
        (mp_vals, "Modus Ponens", COLS["blue"]),
        (ss_vals, "Stat. Syllogism", COLS["orange"]),
        (al_vals, "Both Skills", COLS["green"]),
    ]

    offsets = [-width, 0, width]
    for offset, (vals, skill_name, color) in zip(offsets, bar_specs):
        ax.bar(
            x + offset,
            vals,
            width,
            label=skill_name,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    # ── Gridlines ──────────────────────────────────────────────────────────────
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Group separators & labels ───────────────────────────────────────────────
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes [0,1]

    for xb in separators:
        ax.axvline(xb, color="#AAAAAA", linewidth=1.2, linestyle="--", alpha=0.8, zorder=4)

    for g_start, g_end, g_label in groups:
        center = (g_start + g_end) / 2
        left = g_start - 0.5 + 0.05
        right = g_end + 0.5 - 0.05
        # label above plot
        ax.annotate(
            g_label,
            xy=(center, 1.04),
            xycoords=trans,
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE - 1,
            # weight="bold" if g_label == groups[-1][2] else "normal",  # bold the last group for emphasis
        )
        # bracket line
        ax.annotate(
            "",
            xy=(right, 1.02),
            xycoords=trans,
            xytext=(left, 1.02),
            textcoords=trans,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=1.2),
        )

    # ── Axes ───────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Accuracy", fontsize=AXIS_FONTSIZE)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLS["blue"], label="Modus Ponens"),
        mpatches.Patch(color=COLS["orange"], label="Stat. Syllogism"),
        mpatches.Patch(color=COLS["green"], label="Both Skills"),
    ]
    format_fig(fig)
    leg = ax.legend(
        handles=legend_patches,
        fontsize=TICK_FONTSIZE - 3,
        loc="upper left",
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main() -> None:
    args = parse_args()
    if args.entity is None:
        raise ValueError("WandB entity is required. Set WANDB_ENTITY or pass --entity.")
    # Collect combine-two results from WandB.
    df = collect_results(entity=args.entity, project=args.project)
    if len(df) == 0:
        raise ValueError("No usable runs were found for combine-two evaluation.")

    # Keep only the evaluated metrics we care about for the publication plot.
    metrics = ["statistical syllogism", "modus ponens", "all_skills"]
    df_skills = df[df["skill"].isin(metrics)].copy()

    # Build and save the publication-ready figure.
    if len(df_skills) == 0:
        raise ValueError("No skill evaluation rows available after filtering.")

    fig = skillmix_combine_two(df_skills)
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.abspath(os.path.join(args.output_dir, "skillmix_combine_two.pdf"))
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Saved PDF figure to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
