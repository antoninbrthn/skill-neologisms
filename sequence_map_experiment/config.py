import os
from pathlib import Path

os.environ.setdefault("WANDB_PROJECT", "skill-neologisms")
os.environ.setdefault("WANDB_LOG_MODEL", "false")

ROOT_PATH = str(Path(__file__).resolve().parent)
OUTPUT_DIR = os.path.join(ROOT_PATH, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_PATH, "results")
FIGS_DIR = os.path.join(ROOT_PATH, "figs")
TABLES_DIR = os.path.join(ROOT_PATH, "tables")
TOKENIZER_DIR = os.path.join(ROOT_PATH, "custom_tokenizer")
