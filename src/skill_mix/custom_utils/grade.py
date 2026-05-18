"""
Grading utilities for SkillMix evaluation.
Uses an API model (e.g. GPT-4o) to grade student completions against a rubric.
"""

import re
from typing import List, Dict

from src.models.api_models import OpenAIModel, MODEL_TO_NAME
from src.skill_mix.custom_utils.generate import generate_batch_gpt


def load_grader(grader_model: str, use_batch: bool = True, max_tokens: int = 1024) -> OpenAIModel:
    """Load an API grading model."""
    model_name = MODEL_TO_NAME.get(grader_model, grader_model)
    return OpenAIModel(model_name=model_name, use_batch=use_batch, max_tokens=max_tokens)


def grade_completions(grader: OpenAIModel, grading_prompts: List[str], batch_size: int = 32) -> List[str]:
    """Send grading prompts to the API model and return raw grading outputs.

    Args:
        grader: The grading model
        grading_prompts: List of prompts to grade
        batch_size: Number of prompts to process per batch

    Returns:
        List of raw grading outputs
    """
    return generate_batch_gpt(grader, grading_prompts, batch_size=batch_size)


def parse_grading_output(output: str, num_skills: int) -> Dict:
    """Parse a rubric-style grading output into structured scores.
    Expected format: markdown table with Criteria | Points Earned, ending with Total.
    """
    result = {"raw": output, "score": 0.0, "extracted_score": 0.0, "per_criteria": {}}

    # try table parsing
    lines = output.split("\n")
    table_lines = [l for l in lines if "|" in l and "--" not in l and "criteria" not in l.lower()]

    for line in table_lines:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        key = parts[0].lower().strip()
        val_str = re.sub(r"[^0-9./]", "", parts[1])
        # extract numerator from "X/Y" format
        match = re.search(r"^(\d+)", val_str)
        if not match:
            continue
        val = float(match.group(1))

        if "total" in key:
            result["score"] = val
        else:
            result["per_criteria"][key] = val

    result["extracted_score"] = sum(result["per_criteria"].values())

    # fallback: try "Grade: X" pattern
    if result["score"] == 0.0 and result["extracted_score"] == 0.0:
        grade_match = re.search(r"grade:\s*(\d+(?:\.\d+)?)", output.lower())
        if grade_match:
            result["score"] = float(grade_match.group(1))
            result["extracted_score"] = result["score"]

    total_possible = num_skills + 3  # skills + topic + makes sense + sentence limit
    result["total_possible"] = total_possible
    result["normalized_score"] = result["extracted_score"] / total_possible if total_possible > 0 else 0.0

    return result


def grade_all(
    grader: OpenAIModel,
    records: List[Dict],
    skills_dict: Dict,
    prompt_version: str = "gpt",
    batch_size: int = 32,
) -> List[Dict]:
    """Grade a list of generation records. Adds grading fields in-place and returns them.

    Args:
        grader: The grading model
        records: List of generation records to grade
        skills_dict: Dictionary mapping skill names to descriptions
        prompt_version: Version of grading prompt to use
        batch_size: Number of records to grade per batch

    Returns:
        List of graded records
    """
    from src.skill_mix.custom_utils.data import build_grading_prompt

    grading_prompts = []
    for rec in records:
        skills_str = ", ".join(rec["skills"])
        gp = build_grading_prompt(skills_str, rec["topic"], rec["completion"], skills_dict, prompt_version)
        grading_prompts.append(gp)

    print(f"Grading {len(records)} records with {grader.model_name} (batch_size={batch_size})...")
    raw_outputs = grade_completions(grader, grading_prompts, batch_size=batch_size)
    if isinstance(raw_outputs, str):
        raw_outputs = [raw_outputs]

    for rec, raw in zip(records, raw_outputs):
        parsed = parse_grading_output(raw, len(rec["skills"]))
        rec["grading_raw"] = raw
        rec["score"] = parsed["score"]
        rec["extracted_score"] = parsed["extracted_score"]
        rec["normalized_score"] = parsed["normalized_score"]
        rec["total_possible"] = parsed["total_possible"]
        rec["per_criteria"] = parsed["per_criteria"]

    return records


if __name__ == "__main__":
    # Demo: parse a sample grading output
    sample = """Here's the grading table:

| Criteria | Points Earned |
|---|---|
| Correctly illustrates metaphor | 1 |
| Correctly illustrates modus ponens | 0 |
| Pertains to Gardening | 1 |
| Text makes sense | 1 |
| At most two sentences | 1 |
| Total Points Earned | 4 |

Explanation: The text uses a metaphor but does not illustrate modus ponens."""

    parsed = parse_grading_output(sample, num_skills=2)
    print(f"Score: {parsed['score']}")
    print(f"Extracted: {parsed['extracted_score']}")
    print(f"Normalized: {parsed['normalized_score']:.2f}")
    print(f"Per criteria: {parsed['per_criteria']}")
