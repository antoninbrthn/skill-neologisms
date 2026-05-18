import os
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import wandb
from omegaconf import OmegaConf

from src.config import PROJECT_ROOT


def _resolve_wandb_entity(cfg: OmegaConf = None) -> str:
    if cfg is not None:
        entity = cfg.wandb.get("entity", None)
        if entity:
            return entity
    return os.getenv("WANDB_ENTITY")


def init_wandb(
    cfg: OmegaConf,
    run_formula="{cfg.skill.name}_{cfg.skill.prompt_name}_hf_{timestamp}",
):
    wandb_run = None
    if not cfg.wandb.enabled:
        return wandb_run
    from datetime import datetime

    # Generate run name if not provided
    run_name = cfg.wandb.get("name", None)
    if run_name is None:
        if run_formula is None:
            run_name = "default"
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = run_formula.format(cfg=cfg, timestamp=timestamp)

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=_resolve_wandb_entity(cfg),
        name=run_name,
        tags=cfg.wandb.get("tags", None),
        notes=cfg.wandb.get("notes", ""),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"\nWandB run initialized: {wandb_run.name}")
    print(f"WandB URL: {wandb_run.url}")
    return wandb_run


def load_runs(run_tag=None, project_name="skill_tokens", entity: str = None):
    """Load runs based on a filter, download artifacts, and return run names."""
    entity = entity or os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WandB entity is required. Set WANDB_ENTITY or pass entity=...")

    api = wandb.Api()
    if isinstance(run_tag, str):
        runs = api.runs(f"{entity}/{project_name}", filters={"tags": run_tag})
    elif isinstance(run_tag, list):
        runs = api.runs(f"{entity}/{project_name}", filters={"tags": {"$in": run_tag}})
    else:
        runs = api.runs(f"{entity}/{project_name}")

    runs_dict = {}
    for run in runs:
        runs_dict[run.name + "-" + run.id] = run
    return runs_dict


def get_cache_paths(tags: List[str], project_name: str) -> Tuple[str, str]:
    """Generate cache file paths based on tags.

    Args:
        tags: List of wandb tags
        project_name: WandB project name

    Returns:
        Tuple of (configs_cache_path, results_cache_path)
    """
    cache_dir = os.path.join(PROJECT_ROOT, "exports", "wandb_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique identifier from tags
    tags_str = "_".join(sorted(tags)) if isinstance(tags, list) else tags
    tags_str = tags_str.replace("/", "_").replace(" ", "_")

    configs_cache = os.path.join(cache_dir, f"{project_name}_{tags_str}_configs.pkl")
    results_cache = os.path.join(cache_dir, f"{project_name}_{tags_str}_results.pkl")

    return configs_cache, results_cache


def extract_config_dataframe(runs_dict: Dict) -> pd.DataFrame:
    """Extract configs from runs into a dataframe.

    Args:
        runs_dict: Dictionary of run_name -> run object

    Returns:
        DataFrame with configs indexed by run_id
    """
    config_rows = []

    for run_name, run in runs_dict.items():
        config_data = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "created_at": run.created_at,
            "tags": ",".join(run.tags) if run.tags else "",
        }

        # Flatten config
        config = run.config
        for key, value in config.items():
            if isinstance(value, dict):
                # Flatten nested dicts
                for subkey, subvalue in value.items():
                    config_data[f"{key}.{subkey}"] = subvalue
            else:
                config_data[key] = value

        config_rows.append(config_data)

    df_configs = pd.DataFrame(config_rows)
    df_configs = df_configs.set_index("run_id")

    return df_configs


def extract_results_dataframe(runs_dict: Dict) -> pd.DataFrame:
    """Extract all results from runs into a dataframe.

    Args:
        runs_dict: Dictionary of run_name -> run object

    Returns:
        DataFrame with all run histories concatenated
    """
    results_rows = []

    for run_name, run in runs_dict.items():
        # Get run history (all logged metrics over time)
        history = run.history()

        if len(history) == 0:
            continue

        # Add run metadata to each row
        history["run_id"] = run.id
        history["run_name"] = run.name
        history["run_state"] = run.state
        history["tags"] = ",".join(run.tags) if run.tags else ""

        # Add summary metrics
        summary = run.summary._json_dict
        for key, value in summary.items():
            if key not in history.columns and not isinstance(value, (dict, list)):
                history[f"summary.{key}"] = value

        results_rows.append(history)

    if len(results_rows) == 0:
        return pd.DataFrame()

    df_results = pd.concat(results_rows, ignore_index=True)

    return df_results


def load_wandb_data(tags: List[str], project_name: str = "skill-neologisms", reload_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load wandb runs with caching support.

    Args:
        tags: List of wandb tags to filter runs
        project_name: WandB project name
        reload_cache: If True, reload from wandb API even if cache exists

    Returns:
        Tuple of (df_configs, df_results)
    """
    configs_cache, results_cache = get_cache_paths(tags, project_name)

    # Check if cache exists and should be used
    if not reload_cache and os.path.exists(configs_cache) and os.path.exists(results_cache):
        print(f"Loading from cache...")
        print(f"  Configs: {configs_cache}")
        print(f"  Results: {results_cache}")

        with open(configs_cache, "rb") as f:
            df_configs = pickle.load(f)
        with open(results_cache, "rb") as f:
            df_results = pickle.load(f)

        print(f"Loaded {len(df_configs)} configs and {len(df_results)} result rows from cache")
        return df_configs, df_results

    # Load from wandb API
    print(f"Loading runs from WandB...")
    print(f"  Project: {project_name}")
    print(f"  Tags: {tags}")

    runs_dict = load_runs(run_tag=tags, project_name=project_name)
    print(f"Found {len(runs_dict)} runs")

    # Extract configs
    print("Extracting configs...")
    df_configs = extract_config_dataframe(runs_dict)
    print(f"Extracted {len(df_configs)} configs")

    # Extract results
    print("Extracting results...")
    df_results = extract_results_dataframe(runs_dict)
    print(f"Extracted {len(df_results)} result rows")

    # Save to cache
    print(f"Saving to cache...")
    with open(configs_cache, "wb") as f:
        pickle.dump(df_configs, f)
    with open(results_cache, "wb") as f:
        pickle.dump(df_results, f)
    print(f"  Configs: {configs_cache}")
    print(f"  Results: {results_cache}")

    return df_configs, df_results
