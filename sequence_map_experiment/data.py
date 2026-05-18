from typing import List, Union, Tuple
import random

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

ALL_OPS = {
    # Pretraining operations
    "[ASC]": lambda s: "".join(sorted(s)),
    "[DESC]": lambda s: "".join(sorted(s, reverse=True)),
    "[ADD]": lambda s: "".join(str((int(d) + 1) % 10) for d in s),
    "[SUB]": lambda s: "".join(str((int(d) - 1) % 10) for d in s),
    "[POLARITY]": lambda s: "".join("0" if int(d) % 2 == 0 else "1" for d in s),
    "[REVERSE]": lambda s: s[::-1],
    "[ID]": lambda s: s,
    # New operations to learn
    "[SHIFT_RIGHT]": lambda s: s[-1] + s[:-1],
    "[INVERT_POLARITY]": lambda s: "".join("1" if int(d) % 2 == 0 else "0" for d in s),  # opposite of POLARITY
}

PRETRAIN_OPS = [
    "[ASC]",
    "[DESC]",
    "[ADD]",
    "[SUB]",
    "[POLARITY]",
    "[REVERSE]",
    "[ID]",
]


def extend_ops(sample_ops, task_token_length, pretrain_only=True):
    """
    Extend operation tokens for task token length > 1.
    Eg: ["[ADD]", "[ASC]"] with task_token_length=2 becomes ["<|[ADD]-0|>", "<|[ADD]-1|>", "<|[ASC]-0|>", "<|[ASC]-1|>"]
    """
    # extend operation tokens
    extended_ops = []
    for op in sample_ops:
        if pretrain_only and op not in PRETRAIN_OPS:  # other ops will be extended based on skilltokenmodel logic
            extended_ops.append(op)
            continue
        for t in range(task_token_length):
            extended_ops.append(f"<|{op}-{t}|>")
    sample_ops = extended_ops
    return sample_ops


def generate_sample_data(
    num_samples: int = 1000,
    ops=None,
    seq_len: Union[int, Tuple[int]] = 8,
    min_ops: int = 1,
    max_ops: int = 1,
    reject_len=None,
    reject_ops=None,
    task_token_length: int = 1,
    output_token: str = "=",
) -> List[str]:
    """
    Generate a dataset of mappings with randomly selected operations applied to random digit sequences.

    """
    if type(seq_len) is int:
        seq_len = (seq_len, seq_len)
    assert (
        type(seq_len) is tuple and len(seq_len) == 2 and type(seq_len[0]) is int and type(seq_len[1]) is int
    ), f"seq_len should be int or tuple of (min_len, max_len). Got: {seq_len}"
    if reject_len is None:
        reject_len = []  # eg: [5, 7, 9]
    if reject_ops is None:
        reject_ops = []  # eg: ["[ASC]", "[ASC][DESC]", "[ADD][POLARITY]"]
    if ops is None:
        ops = PRETRAIN_OPS
    data = []
    i = 0
    itr = 0
    max_iter = num_samples * 100  # to avoid infinite loop
    while i < num_samples or itr >= max_iter:
        itr += 1
        sampled_len = random.randint(seq_len[0], seq_len[1])
        if sampled_len in reject_len:
            continue  # reject

        # Sample sequence of len sampled_len
        seq = "".join(str(random.randint(0, 9)) for _ in range(sampled_len))
        # Sample operation
        sample_ops = random.sample(list(ops), random.randint(min_ops, max_ops))
        if "".join(sample_ops) in reject_ops:
            continue  # reject
        result = seq
        for op in sample_ops:
            result = ALL_OPS[op](result)
        if task_token_length > 1:
            sample_ops = extend_ops(sample_ops, task_token_length)

        data.append(f"{''.join(sample_ops)}{seq}{output_token}{result}")
        i += 1
    return data


def generate_sample_data_skill(
    main_ops: list,
    other_ops: list = None,
    num_samples: int = 1000,
    seq_len: int = 8,
    min_ops: int = 1,
    max_ops: int = 1,
    reject_len=None,
    reject_ops=None,
    task_token_length: int = 1,
    output_token: str = "=",
    corrupt_ops_ratio: float = 0,
) -> List[str]:
    """
    Generates samples using a specific operation and optionally other operations for compositions.
    The main ops will aways be included in the operation sequence. If max_ops > 1, other ops will be sampled to fill the rest.
    Inputs:
        main_ops: List of main operations to always include (e.g., ["[SHIFT_RIGHT]"])
        other_ops: List of other operations to sample from (e.g., ["[ASC]", "[ADD]"])
        num_samples: Number of samples to generate
        seq_len: Length of the digit sequence
        min_ops: Minimum number of operations in the sequence
        max_ops: Maximum number of operations in the sequence
        reject_len: List of sequence lengths to reject
        reject_ops: List of operation sequences to reject
        task_token_length: Length of task tokens (for extending tokens)
        corrupt_ops_ratio: Ratio of samples in which the main operation will be replaced with a random other operation (for ablation experiment on noisy skill labels)
    Returns:
        List of generated samples as strings
    """
    if type(seq_len) is int:
        seq_len = (seq_len, seq_len)
    if reject_len is None:
        reject_len = []  # eg: [5, 7, 9]
    if reject_ops is None:
        reject_ops = []  # eg: ["[ASC]", "[ASC][DESC]", "[ADD][POLARITY]"]

    data = []
    i = 0
    itr = 0
    max_iter = num_samples * 100  # to avoid infinite loop
    while i < num_samples or itr >= max_iter:
        itr += 1
        sampled_len = random.randint(seq_len[0], seq_len[1])
        if sampled_len in reject_len:
            continue  # reject

        # Sample sequence
        seq = "".join(str(random.randint(0, 9)) for _ in range(sampled_len))
        # Sample operations
        num_ops = random.randint(min_ops, max_ops)
        if num_ops == 1:
            sample_ops = [random.choice(main_ops)]
        else:
            sample_ops = random.sample(list(other_ops), num_ops - 1)
            sample_ops.append(random.choice(main_ops))
            random.shuffle(sample_ops)

        if "".join(sample_ops) in reject_ops:
            continue  # reject

        result = seq
        for op in sample_ops:
            # for ablation experiment (noisy skill labels)
            if (op in main_ops) and (corrupt_ops_ratio > 0) and (random.random() < corrupt_ops_ratio):
                op = random.choice(other_ops)
            result = ALL_OPS[op](result)

        if task_token_length > 1:
            sample_ops = extend_ops(sample_ops, task_token_length)

        data.append(f"{''.join(sample_ops)}{seq}{output_token}{result}")
        i += 1
    return data


def split_on_output(data, output_token: str = "=") -> Tuple[List[str], List[str]]:
    prompts = [t.split(output_token)[0] + output_token for t in data]
    labels = [t.split(output_token)[1] for t in data]
    return prompts, labels


def count_right_pad(x, pad_id):
    # x shape (L,)
    n = 0
    for t in reversed(x.tolist()):
        if t == pad_id:
            n += 1
        else:
            break
    return n


def apply_random_skill_padding(input_ids, attention_mask, tokenizer, max_length):
    """
    Apply random padding around operation tokens.
    """
    # check shape is (1,L)
    # if (k, L), iterate over it
    if input_ids.shape[0] > 1:
        for i in range(input_ids.shape[0]):
            input_ids[i : i + 1], attention_mask[i : i + 1] = apply_random_skill_padding(
                input_ids[i : i + 1],
                attention_mask[i : i + 1],
                tokenizer,
                max_length,
            )
        return input_ids, attention_mask

    pad_id = tokenizer.pad_token_id

    # Get operation token ids
    op_token_ids = [tokenizer.convert_tokens_to_ids(op) for op in ALL_OPS.keys()]

    # Convert to list for easier manipulation
    input_ids_list = input_ids.squeeze().tolist()
    attention_mask_list = attention_mask.squeeze().tolist()

    # Find all operation token positions (process from right to left to preserve indices)
    op_positions = []
    for i, token_id in enumerate(input_ids_list):
        if token_id in op_token_ids:
            op_positions.append(i)

    # Process from right to left to preserve indices during insertion
    for pos in reversed(op_positions):
        # Sample N: 0 with prob 0.7, or uniformly between 1-20 with prob 0.3
        if random.random() < 0.7:
            N = 0
        else:
            N = random.randint(1, 20)
        if N > 0:
            # Split N into M (left padding) and N-M (right padding) randomly
            M = random.randint(0, N)
            right_pad = N - M
            # Insert M pad tokens before the operation token
            for _ in range(M):
                input_ids_list.insert(pos, pad_id)
                attention_mask_list.insert(pos, 0)

            # Insert right_pad tokens after the operation token (accounting for M insertions)
            for _ in range(right_pad):
                input_ids_list.insert(pos + M + 1, pad_id)
                attention_mask_list.insert(pos + M + 1, 0)

    # Truncate to max_length if needed
    # assert all the truncated tokens are pad tokens
    assert all(tid == pad_id for tid in input_ids_list[max_length:]), "Non-pad tokens found in truncated part."
    input_ids_list = input_ids_list[:max_length]
    attention_mask_list = attention_mask_list[:max_length]

    # Convert back to tensors with original shape
    input_ids = torch.tensor(input_ids_list).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask_list).unsqueeze(0)

    return input_ids, attention_mask


class SequenceTaskDataset(Dataset):
    """Dataset for sequence transformation tasks."""

    def __init__(
        self,
        data: List[str],
        tokenizer,
        max_length: int = 128,
        random_left_pad=False,
        random_skill_pad=False,
        output_token="=",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_left_pad = random_left_pad  # if true, will pad on left randomly with whatever length is left
        self.random_skill_pad = random_skill_pad  # if true, will pad around operation tokens randomly
        self.output_token = output_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        if (self.output_token != "[OUTPUT]") and "[OUTPUT]" in text:
            print(f"Warning: Replacing [OUTPUT] with {self.output_token} in dataset text.")
            text = text.replace("[OUTPUT]", self.output_token) if type(text) is str else [t.replace("[OUTPUT]", self.output_token) for t in text]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]  # (1, L)
        attention_mask = encoding["attention_mask"]

        if self.random_skill_pad:
            # input_ids and attention_mask should be shape (1, L)
            # if shape[0] is >1, iterate over it
            input_ids, attention_mask = apply_random_skill_padding(input_ids, attention_mask, self.tokenizer, self.max_length)

            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        if self.random_left_pad:
            # take a random number of the pad tokens from the left side
            pad_id = self.tokenizer.pad_token_id  # or 10

            for i in range(input_ids.shape[0]):
                right_pad = count_right_pad(input_ids[i], pad_id)
                k = torch.randint(0, right_pad + 1, (1,)).item()
                if k > 0:
                    input_ids[i] = torch.cat([torch.full((k,), pad_id), input_ids[i][:-k]])
                    attention_mask[i] = torch.cat(
                        [
                            torch.zeros(k, dtype=attention_mask.dtype),
                            attention_mask[i][:-k],
                        ]
                    )
            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Create labels: -100 for input part (not trained), actual tokens for output
        labels = input_ids.clone()

        # Find [OUTPUT] token position
        output_token_id = self.tokenizer.convert_tokens_to_ids(self.output_token)
        output_positions = (input_ids == output_token_id).nonzero(as_tuple=True)[-1]
        if len(labels.shape) == 1:
            assert len(output_positions) == 1, "There should be exactly one [OUTPUT] token per sequence."
            labels[: output_positions[0].item() + 1] = -100
        else:
            for i, pos in enumerate(output_positions):
                # Mask everything before the [OUTPUT] token itself
                labels[i, : pos.item() + 1] = -100

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def generate_train_val_datasets(
    cfg,
    skill_op,
    test_op,
    other_ops=None,
    num_train_samples=100_000,
    num_val_samples=5000,
    task_token_length=1,
    output_token="=",
):
    """
    Generate training and validation datasets for 1 to k operations.

    Args:
        cfg: Configuration object
        skill_op: The skill operation (e.g., "[SHIFT_RIGHT]")
        test_op: The test operation to exclude (e.g., "[ADD]")
        num_train_samples: Number of training samples per operation count
        num_val_samples: Number of validation samples per operation count

    Returns:
        tuple: (train_data_dict, val_data_dict) where keys are operation counts (1, 2, ..., max_ops)
    """
    # Get all operations except test_op
    if other_ops is None:
        other_ops = [op for op in PRETRAIN_OPS if op != test_op]

    seq_len = cfg.dataset.get("seq_length", 8)
    seq_len_min = cfg.dataset.get("seq_len_min", seq_len)
    reject_len = cfg.dataset.get("reject_len", None)
    reject_ops = cfg.dataset.get("reject_ops", None)
    max_ops = cfg.dataset.get("max_ops", 2)
    corrupt_ops_ratio = cfg.dataset.get("corrupt_ops_ratio", 0.0)  # ablation experiment

    print(f"\nGenerating train/val datasets:")
    print(f"  Skill op: {skill_op}")
    print(f"  Test op (excluded): {test_op}")
    print(f"  Other ops: {other_ops}")
    print(f"  Seq length: {seq_len_min}-{seq_len}")
    print(f"  Max ops: {max_ops}")
    print(f"  Output token: {output_token}")

    train_data_dict = {}
    val_data_dict = {}

    # Generate data for each operation count from 1 to max_ops
    num_train_samples_per_op = num_train_samples // max_ops
    num_val_samples_per_op = num_val_samples // max_ops
    for num_ops in range(1, max_ops + 1):
        # Generate training data
        print(f"\nGenerating {num_ops}-op training data ({num_train_samples_per_op} samples)...")
        train_data_dict[num_ops] = generate_sample_data_skill(
            main_ops=[skill_op],
            other_ops=other_ops,
            num_samples=num_train_samples_per_op,
            min_ops=num_ops,
            max_ops=num_ops,
            seq_len=(seq_len_min, seq_len),
            reject_len=reject_len,
            reject_ops=reject_ops,
            task_token_length=task_token_length,
            output_token=output_token,
            corrupt_ops_ratio=corrupt_ops_ratio,
        )

        # Generate validation data
        print(f"Generating {num_ops}-op validation data ({num_val_samples_per_op} samples)...")
        val_data_dict[num_ops] = generate_sample_data_skill(
            main_ops=[skill_op],
            other_ops=other_ops,
            num_samples=num_val_samples_per_op,
            min_ops=num_ops,
            max_ops=num_ops,
            seq_len=(seq_len_min, seq_len),
            reject_len=reject_len,
            reject_ops=reject_ops,
            task_token_length=task_token_length,
            output_token=output_token,
            # corrupt_ops_ratio=corrupt_ops_ratio,  # validate on clean data for ablation experiment
        )

    # show random 3 samples from train set
    for _ in range(3):
        sample_num_ops = random.choice(list(train_data_dict.keys()))
        sample_data = train_data_dict[sample_num_ops]
        sample_example = random.choice(sample_data)
        print(f"  Sample from {sample_num_ops}-op train set: {sample_example}")
    return train_data_dict, val_data_dict


def generate_test_datasets(cfg, skill_op, test_op, num_samples=1000, task_token_length=1, output_token="="):
    """
    Generate test datasets for k-op combinations that include test_op.
    For each k from 2 to max_ops, generates all permutations of k operations
    where exactly 1 is test_op, exactly 1 is skill_op, and (k-2) are other_ops.

    Args:
        cfg: Configuration object
        skill_op: The skill operation
        test_op: The test operation to test with
        num_samples: Number of samples per combination

    Returns:
        dict: Nested dictionary test_datasets[num_ops][seq_len][permutation_key] = data_list
              permutation_key uses 'X' to denote other_ops (e.g., 'test_X_skill', 'X_skill_test')
    """
    from itertools import permutations, product
    from sequence_map_experiment.data import ALL_OPS

    seq_len = cfg.dataset.get("seq_length", 8)
    seq_len_min = cfg.dataset.get("seq_len_min", seq_len)
    max_ops = cfg.dataset.get("max_ops", 2)
    seq_lengths = list(range(seq_len_min, seq_len + 1))

    # Get all operations except test_op
    other_ops = [op for op in PRETRAIN_OPS if op != test_op]

    print(f"\nGenerating test datasets:")
    print(f"  Skill op: {skill_op}")
    print(f"  Test op: {test_op}")
    print(f"  Other ops: {other_ops}")
    print(f"  Max ops: 3 (model trained on {max_ops})")
    print(f"  Seq lengths: {seq_lengths}")
    print(f"  Task_token_length: {task_token_length}")
    print(f"  Output token: {output_token}")

    test_datasets = {}

    # For each operation count from 2 to max_ops
    num_samples_per_combination = num_samples // (len(seq_lengths) * max_ops)

    for num_ops in range(2, 4):  # test up to 3-ops
        test_datasets[num_ops] = {}

        # For k operations: 1 test_op, 1 skill_op, and (k-2) other_ops
        num_other_ops = num_ops - 2

        if num_other_ops == 0:
            # For 2 operations: just test_op and skill_op
            ops_combinations = [()]
        else:
            # Generate all combinations of other_ops to fill (k-2) spots
            # We can have repeated operations from other_ops
            ops_combinations = list(product(other_ops, repeat=num_other_ops))

        print(f"\n  {num_ops}-op combinations: {len(ops_combinations)} other_op combinations × permutations")

        for slen in seq_lengths:
            test_datasets[num_ops][slen] = {}

            for other_ops_tuple in ops_combinations:
                # Create the full list: test_op, skill_op, and other_ops
                ops_list = [test_op, skill_op] + list(other_ops_tuple)

                # Get all unique permutations
                unique_perms = list(set(permutations(ops_list)))

                for op_perm in unique_perms:
                    # Create permutation key using 'X' for other_ops
                    perm_key_parts = []
                    for op in op_perm:
                        if op == test_op:
                            perm_key_parts.append("test")
                        elif op == skill_op:
                            perm_key_parts.append("skill")
                        else:
                            perm_key_parts.append("X")
                    perm_key = "_".join(perm_key_parts)

                    # Skip if this permutation key already exists (multiple other_ops can create same pattern)
                    if perm_key in test_datasets[num_ops][slen]:
                        continue

                    test_data = []
                    for _ in range(num_samples_per_combination):
                        seq = "".join(str(torch.randint(0, 10, (1,)).item()) for _ in range(slen))

                        # Apply operations in sequence
                        result = seq
                        for op in op_perm:
                            result = ALL_OPS[op](result)
                        if task_token_length > 1:
                            op_perm_ext = extend_ops(op_perm, task_token_length)
                        else:
                            op_perm_ext = op_perm
                        # Create prompt with all operations
                        ops_str = "".join(op_perm_ext)
                        test_data.append(f"{ops_str}{seq}{output_token}{result}")

                    test_datasets[num_ops][slen][perm_key] = test_data

            print(f"    Generated {len(test_datasets[num_ops][slen])} unique permutation patterns for seq_len={slen}")

    if len(test_datasets) == 0:
        print(f"  No test datasets generated (max_ops={max_ops} < 2).")
        return test_datasets

    # show random 3 samples from one of the datasets
    for _ in range(3):
        sample_num_ops = random.choice(list(test_datasets.keys()))
        sample_slen = random.choice(seq_lengths)
        sample_perm_key = random.choice(list(test_datasets[sample_num_ops][sample_slen].keys()))
        sample_data = test_datasets[sample_num_ops][sample_slen][sample_perm_key]
        sample_example = random.choice(sample_data)
        print(f"  Sample from {sample_num_ops}-op, seq_len={sample_slen}, perm={sample_perm_key}: {sample_example}")
    return test_datasets
