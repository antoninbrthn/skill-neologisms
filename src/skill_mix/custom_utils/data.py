"""
Data utilities for SkillMix evaluation.
Handles loading skills/topics and generating prompt combinations.
"""

import csv
import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional

from src.config import SKILL_MIX_ROOT


def load_skills(csv_path: Optional[str] = None) -> Dict[str, Dict]:
    csv_path = csv_path or os.path.join(SKILL_MIX_ROOT, "skills.csv")
    skills = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ["remove", "in review", "maybe"]:
                row["skill"] = row["skill"].strip()
                skills[row["skill"]] = row
    return skills


def load_topics(txt_path: Optional[str] = None) -> List[str]:
    txt_path = txt_path or os.path.join(SKILL_MIX_ROOT, "topics.txt")
    topics = Path(txt_path).read_text().strip().split("\n")
    return [t.strip() for t in topics if t.strip()]


def make_combinations_with_anchor_skills(
    skills: List[str],
    anchor_skills: List[str],
    k: int,
    num_combinations: int = 10,
    seed: int = 0,
) -> List[List[str]]:
    rng = random.Random(seed)

    # Every combination should have all anchor_skills and k-len(anchor_skills) from skills
    # Random sampling
    result = []
    for _ in range(num_combinations):
        if len(anchor_skills) >= k:  # if anchor_skills already >= k, just sample from anchor_skills
            rng.shuffle(anchor_skills)
            combo = anchor_skills[:k]
            result.append(combo)
        else:  # sample from non-anchor skills to fill up to k
            available_skills = [s for s in skills if s not in anchor_skills]
            rng.shuffle(available_skills)
            combo = anchor_skills + available_skills[: k - len(anchor_skills)]
            rng.shuffle(combo)
            result.append(combo)
    return result


def make_combinations(skills: List[str], k: int, num_combinations: int = 10, seed: int = 0) -> List[List[str]]:
    rng = random.Random(seed)
    result = []
    for _ in range(num_combinations):
        combo = list(skills)
        rng.shuffle(combo)
        result.append(combo[:k])
    return result


def assign_topics(
    combos: List[List[str]],
    topics: List[str],
    num_topics: int = 1,
    fix_topics: bool = False,
    seed: int = 0,
) -> List[List[str]]:
    """For each combo, assign `num_topics` random topics. Returns parallel list of topic-lists."""
    rng = random.Random(seed)
    if fix_topics:
        shuffled = list(topics)
        rng.shuffle(shuffled)
        fixed = shuffled[:num_topics]
        return [fixed] * len(combos)
    result = []
    for _ in combos:
        shuffled = list(topics)
        rng.shuffle(shuffled)
        result.append(shuffled[:num_topics])
    return result


def build_generation_prompt(
    skills_list: List[str],
    topic: str,
    skills_dict: Dict[str, Dict],
    prompt_version: str = "default",
) -> str:
    """Build the first-turn generation prompt from the JSON template."""
    prompt_file = os.path.join(SKILL_MIX_ROOT, "prompts", "generation", f"{prompt_version}.json")
    with open(prompt_file) as f:
        templates = json.load(f)

    # only use first template (single turn)
    template = templates[0]

    skills_str = ", ".join(skills_list)
    num_skills = len(skills_list)

    def _fmt(skill, style="simple"):
        d = skills_dict[skill]
        if style == "simple":
            return f"**{skill}**: {d['definition']} For example, {d['example']}"
        return f"Skill: {skill}\nDefinition: {d['definition']}\nExample: {d['example']}"

    skills_defs_and_examples = "\n".join(_fmt(s, "regular") for s in skills_list)
    skills_defs_and_examples_simple = "\n".join(_fmt(s, "simple") for s in skills_list)
    skills_defs = "\n".join(f"**{s}**: {skills_dict[s]['definition']}" for s in skills_list)

    num_sentences = max(1, num_skills - 1)
    import inflect

    p = inflect.engine()
    num_skills_str = p.number_to_words(num_skills)
    num_sentences_str = p.number_to_words(num_sentences) + " " + ("sentences" if num_sentences > 1 else "sentence")

    prompt = template.format(
        skills_str=skills_str,
        skills_defs_and_examples=skills_defs_and_examples,
        skills_defs_and_examples_simple=skills_defs_and_examples_simple,
        skills_defs=skills_defs,
        num_skills=num_skills,
        num_skills_str=num_skills_str,
        topic=topic,
        num_sentences=num_sentences,
        num_sentences_str=num_sentences_str,
    )
    return prompt


def build_grading_prompt(
    skills_str: str,
    topic: str,
    student_answer: str,
    skills_dict: Dict[str, Dict],
    prompt_version: str = "gpt",
) -> str:
    """Build the grading prompt from the JSON template."""
    prompt_file = os.path.join(SKILL_MIX_ROOT, "prompts", "grade", f"{prompt_version}.json")
    with open(prompt_file) as f:
        templates = json.load(f)
    template = templates[0]

    skills_list = [s.strip() for s in skills_str.split(",")]
    num_skills = len(skills_list)

    def _fmt(skill, style="simple"):
        d = skills_dict[skill]
        if style == "simple":
            return f"**{skill}**: {d['definition']} For example, {d['example']}"
        return f"Skill: {skill}\nDefinition: {d['definition']}\nExample: {d['example']}"

    skills_defs_and_examples_simple = "\n".join(_fmt(s, "simple") for s in skills_list)
    skills_defs_and_examples = "\n".join(_fmt(s, "regular") for s in skills_list)
    skills_defs = "\n".join(f"**{s}**: {skills_dict[s]['definition']}" for s in skills_list)

    num_sentences = max(1, num_skills - 1)
    import inflect

    p = inflect.engine()
    num_skills_str = p.number_to_words(num_skills)
    num_sentences_str = p.number_to_words(num_sentences) + " " + ("sentences" if num_sentences > 1 else "sentence")

    # build rubric items
    rubric_skills = [f"Correctly illustrates {s.strip().lower()}" for s in skills_list]
    rubric_items = rubric_skills + [
        f"Pertains to {topic}",
        "Text makes sense",
        f"At most {num_sentences_str}",
    ]
    rubric_items_str = ", ".join(rubric_items)

    prompt = template.format(
        skills_str=", ".join(skills_list),
        skills_defs_and_examples=skills_defs_and_examples,
        skills_defs_and_examples_simple=skills_defs_and_examples_simple,
        skills_defs=skills_defs,
        student_answer=student_answer,
        num_skills=num_skills,
        num_skills_str=num_skills_str,
        topic=topic,
        num_sentences=num_sentences,
        num_sentences_str=num_sentences_str,
        rubric_items=rubric_items_str,
    )
    return prompt


def generate_eval_dataset(cfg) -> List[Dict]:
    """Generate the list of (skills, topic) eval items from config.
    Returns list of dicts: {'skills': [...], 'topic': str, 'prompt': str}
    """
    sm_cfg = cfg.skill_mix
    skills_dict = load_skills(sm_cfg.get("skills_csv", None))
    topics = load_topics(sm_cfg.get("topics_txt", None))
    skill_names = list(skills_dict.keys())

    use_specific_skills = sm_cfg.get("use_specific_skills", False)
    if use_specific_skills:
        new_skills = list(sm_cfg.get("new_skills", []))
        train_skills = list(sm_cfg.get("train_skills", []))
        test_skills = list(sm_cfg.get("test_skills", []))
        exclude_skills = list(sm_cfg.get("exclude_skills", []))

        # if train_skills is empty, set to all skills except new_skills, test_skills and exclude_skills
        if not train_skills:
            train_skills = [s for s in skill_names if s not in new_skills and s not in test_skills and s not in exclude_skills]
        print(f"Using skills from config: {len(new_skills)} new, {len(train_skills)} train, {len(test_skills)} test")
        print(f"  New skills: {new_skills}")
        print(f"  Train skills: {train_skills}")
        print(f"  Test skills: {test_skills}")
        print(f"  Exclude skills: {exclude_skills}")
        combos = make_combinations_with_anchor_skills(train_skills, new_skills, sm_cfg.k, sm_cfg.num_combinations, sm_cfg.seed)

    else:
        combos = make_combinations(skill_names, sm_cfg.k, sm_cfg.num_combinations, sm_cfg.seed)

    topics_per_combo = assign_topics(combos, topics, sm_cfg.num_topics, sm_cfg.get("fix_topics", False), sm_cfg.seed)

    gen_cfg = cfg.generation
    dataset = []
    for combo, topic_list in zip(combos, topics_per_combo):
        for topic in topic_list:
            prompt = build_generation_prompt(combo, topic, skills_dict, gen_cfg.get("prompt_version", "default"))
            dataset.append(
                {
                    "skills": combo,
                    "topic": topic,
                    "prompt": prompt,
                }
            )
    return dataset


if __name__ == "__main__":
    # Quick demo
    skills_dict = load_skills()
    topics = load_topics()
    print(f"Loaded {len(skills_dict)} skills, {len(topics)} topics")

    combos = make_combinations(list(skills_dict.keys()), k=2, num_combinations=3, seed=42)
    for c in combos:
        print(f"  combo: {c}")

    prompt = build_generation_prompt(combos[0], topics[0], skills_dict)
    print(f"\nSample prompt (first 300 chars):\n{prompt[:300]}...")
