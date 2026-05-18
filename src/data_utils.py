import os
import importlib

from omegaconf import DictConfig, OmegaConf
from datasets import Dataset

from src.config import PROJECT_ROOT
from src.renderer.prompts_renderer import PromptLibrary
from src.trainer_utils import (
    generate_prompt_dataset,
)


def import_function(module_path: str, function_name: str):
    """Dynamically import a function from a module path.

    Args:
        module_path: Python module path (e.g., "my_package.my_module")
        function_name: Name of the function to import

    Returns:
        The imported function
    """
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def generate_dataset_from_config(cfg: DictConfig) -> Dataset:
    """Generate the full dataset based on the configuration."""
    # Check if using new pipeline interface or old function interface
    if hasattr(cfg.dataset.generator, "pipeline") and cfg.dataset.generator.pipeline:
        # New pipeline interface
        data_generator = import_function(cfg.dataset.generator.module, "generate")
        print(f"Loaded data generator: {cfg.dataset.generator.module}.generate (pipeline mode)")

        # Merge kwargs and pipeline into data_generator_kwargs
        kwargs = OmegaConf.to_container(cfg.dataset.generator.kwargs, resolve=True)
        kwargs["pipeline"] = cfg.dataset.generator.pipeline
        data_generator_kwargs = kwargs
    else:
        # Old interface (backward compatible)
        data_generator = import_function(cfg.dataset.generator.module, cfg.dataset.generator.function)
        print(f"Loaded data generator: {cfg.dataset.generator.module}.{cfg.dataset.generator.function}")
        data_generator_kwargs = OmegaConf.to_container(cfg.dataset.generator.kwargs, resolve=True)

    prompt_yaml_path = os.path.join(PROJECT_ROOT, "configs", "tasks", cfg.skill.prompt_file)
    prompt_library = PromptLibrary(prompt_yaml_path)
    print(f"Loaded prompt library from: {prompt_yaml_path}")
    print(f"Using prompt template: {cfg.skill.prompt_name}")

    full_dataset = generate_prompt_dataset(
        data_generator=data_generator,
        prompt_library=prompt_library,
        prompt_name=cfg.skill.prompt_name,
        num_samples=cfg.dataset.total_size,
        data_generator_kwargs=data_generator_kwargs,
        num_examples_range=cfg.dataset.num_examples_range,
        layout_id=cfg.skill.get("layout_id", None),
    )
    return full_dataset
