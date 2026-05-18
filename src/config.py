import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SKILL_MIX_ROOT = os.path.join(PROJECT_ROOT, "src", "skill_mix")
