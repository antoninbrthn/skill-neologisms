from typing import Dict, List, Tuple
import random

from tqdm import tqdm
import pandas as pd

from sequence_map_experiment.data import generate_sample_data_skill, split_on_output
from sequence_map_experiment.utils import evaluate_model_batch


def generate_test_dataset(
    main_ops: List[str],
    other_ops: List[str],
    num_samples: int = 100,
    seq_len_range: Tuple[int, int] = (2, 9),
    min_ops: int = 1,
    max_ops: int = 3,
    reject_len: List[int] = None,
    reject_ops: List[str] = None,
    task_token_length: int = 1,
    output_token: str = "=",
) -> Dict[int, List[str]]:
    reject_len = reject_len or []
    reject_ops = reject_ops or []

    test_datasets = {}
    seq_lengths = list(range(seq_len_range[0], seq_len_range[1] + 1))
    for seq_len in seq_lengths:
        if seq_len in reject_len:
            continue
        dataset = generate_sample_data_skill(
            main_ops=main_ops,
            other_ops=other_ops,
            num_samples=num_samples,
            seq_len=seq_len,
            min_ops=min_ops,
            max_ops=max_ops,
            reject_len=reject_len,
            reject_ops=reject_ops,
            task_token_length=task_token_length,
            output_token=output_token,
        )
        test_datasets[seq_len] = dataset

    return test_datasets


def test_on_skill_1op_varying_lengths(
    adapter,
    cfg,
    num_samples=100,
    device="cuda",
    verbose=True,
    prompt_format_type="skill",
    prefix_name: str = None,
):
    """Test model on individual tasks with varying sequence lengths.

    Args:
        adapter: SkillTokenModel adapter with model and tokenizer
        cfg: Configuration with dataset settings
        num_samples: Number of samples per task/length combination
        device: Device to run on
        verbose: Whether to print progress
        prompt_format_type: Type of prompt formatting ("skill" or "id"). "skill" expands to skill tokens; "id" keeps prompt as is.

    Returns:
        DataFrame with columns: task, seq_len, accuracy, num_correct, num_total
    """
    results = []

    skill_op = [cfg.dataset.skill_op]  # e.g., "[SHIFT_RIGHT]"
    seq_len = cfg.dataset.get("seq_length", 8)
    seq_len_min = cfg.dataset.get("seq_len_min", seq_len)
    seq_lengths = list(range(seq_len_min, seq_len + 1))

    for slen in tqdm(seq_lengths):
        if verbose:
            print(f"\nTesting {skill_op} with seq_len={slen}...")

        raw_test_data = generate_sample_data_skill(
            main_ops=skill_op,
            num_samples=num_samples,
            min_ops=1,
            max_ops=1,
            seq_len=(slen, slen),
        )
        # If a prefix skill is provided, prepend it and expand.
        if prefix_name is not None:
            prefix_tag = f"<|{prefix_name}|>"
            test_data = [prefix_tag + t for t in raw_test_data]
            test_data = [adapter._expand_skill_tokens(t) for t in test_data]
        else:
            if prompt_format_type == "id":  # for baselines - keep prompt as is
                test_data = raw_test_data
            else:  # skill neologisms
                # insert skill tokens in the input prompts
                test_data = [t.replace(skill_op[0], f"<|{skill_op[0]}|>") for t in raw_test_data]
                test_data = [adapter._expand_skill_tokens(t) for t in test_data]

        print(f"  Evaluating on {len(test_data)} samples...")
        print("    Example 2 test samples:")
        for sample in [test_data[i] for i in random.sample(range(len(test_data)), 2)]:
            print(f"      {sample}")

        acc, detailed = evaluate_model_batch(adapter.model, adapter.tokenizer, test_data, device=device, verbose=verbose)
        num_correct = int(acc * len(test_data))

        results.append(
            {
                "task": skill_op,
                "seq_len": slen,
                "accuracy": acc,
                "num_correct": num_correct,
                "num_total": len(test_data),
            }
        )
        if verbose:
            print(f"  Accuracy: {acc:.2%} ({num_correct}/{len(test_data)})")

    return pd.DataFrame(results)
