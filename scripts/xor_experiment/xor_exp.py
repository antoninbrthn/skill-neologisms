"""Section 4 motivating experiment: XOR/XNOR keyword vs text-description prompts."""

import argparse
import gc
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.config import PROJECT_ROOT
from src.models.loading import load_base_hf_model


@dataclass
class PromptVariant:
    name: str
    prefix_template: str


def _int_to_bin(x: int, bit_length: int) -> str:
    return f"{x:0{bit_length}b}"


def _apply_op(a_bits: str, b_bits: str, op: str) -> str:
    a = int(a_bits, 2)
    b = int(b_bits, 2)
    if op == "XOR":
        out = a ^ b
    elif op == "XNOR":
        out = ~(a ^ b) & ((1 << len(a_bits)) - 1)
    else:
        raise ValueError(f"Unsupported op: {op}")
    return _int_to_bin(out, len(a_bits))


def _sample_pair(bit_length: int, rng: random.Random) -> Tuple[str, str]:
    max_val = 2**bit_length - 1
    return _int_to_bin(rng.randint(0, max_val), bit_length), _int_to_bin(rng.randint(0, max_val), bit_length)


def _build_prompt(
    op: str,
    bit_length: int,
    num_shots: int,
    variant: PromptVariant,
    rng: random.Random,
) -> Tuple[str, str]:
    examples = []
    for _ in range(num_shots):
        a, b = _sample_pair(bit_length, rng)
        y = _apply_op(a, b, op)
        examples.append(f"{a} {b} = {y}")

    query_a, query_b = _sample_pair(bit_length, rng)
    expected = _apply_op(query_a, query_b, op)

    prefix = variant.prefix_template.format(op=op.upper())
    prompt = "\n".join([prefix] + examples + [f"{query_a} {query_b} ="])
    return prompt, expected


def _extract_prediction(text: str, bit_length: int) -> str:
    match = re.search(rf"[01]{{{bit_length}}}", text)
    return match.group(0) if match else ""


@torch.inference_mode()
def _evaluate_variant(
    model,
    tokenizer,
    op: str,
    bit_length: int,
    num_shots: int,
    num_samples: int,
    variant: PromptVariant,
    rng: random.Random,
    batch_size: int,
) -> float:
    prompts, labels = [], []
    for _ in range(num_samples):
        p, y = _build_prompt(op=op, bit_length=bit_length, num_shots=num_shots, variant=variant, rng=rng)
        prompts.append(p)
        labels.append(y)

    preds: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_token_type_ids=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        out_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=bit_length + 4,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        gen_only = out_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        preds.extend([_extract_prediction(t, bit_length) for t in decoded])

    return float(np.mean([p == y for p, y in zip(preds, labels)]))


def run_experiment(
    model_names: List[str],
    bit_length: int,
    num_shots: int,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    variants = [
        PromptVariant("Only Examples", ""),
        PromptVariant(
            "Examples + Text Description",
            "Complete the following using the skill: 'output 1 iif both input bits are {text_desc}'",
        ),
        PromptVariant("Examples + Keyword", "Complete the following using the skill: '{op}'"),
    ]

    desc_by_op = {
        "XOR": "different, and 0 otherwise",
        "XNOR": "the same, and 0 otherwise",
    }

    rows = []
    for model_name in model_names:
        print(f"\\nLoading model: {model_name}")
        model, tokenizer, _ = load_base_hf_model(model_name=model_name, eval_mode=True, pad_token="eos")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_rng = random.Random(seed)
        for op in ["XOR", "XNOR"]:
            for variant in variants:
                if "{text_desc}" in variant.prefix_template:
                    prefix = variant.prefix_template.format(text_desc=desc_by_op[op], op=op.upper())
                    eval_variant = PromptVariant(variant.name, prefix)
                else:
                    eval_variant = variant

                acc = _evaluate_variant(
                    model=model,
                    tokenizer=tokenizer,
                    op=op,
                    bit_length=bit_length,
                    num_shots=num_shots,
                    num_samples=num_samples,
                    variant=eval_variant,
                    rng=model_rng,
                    batch_size=batch_size,
                )
                se = float(np.sqrt(acc * (1 - acc) / num_samples))
                rows.append(
                    {
                        "model_name": model_name,
                        "task": op.upper(),
                        "description": variant.name,
                        "accuracy": acc,
                        "accuracy_se": se,
                        "N": num_samples,
                    }
                )
                print(f"  {op.upper()} | {variant.name}: {acc:.3f}")

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df_plot = df.copy()
    model_order = [
        "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-3.1-8B",
        "mistralai/Ministral-3-8B-Base-2512",
        "microsoft/phi-4",
    ]
    desc_order = ["Only Examples", "Examples + Text Description", "Examples + Keyword"]

    df_plot["model_name"] = pd.Categorical(df_plot["model_name"], categories=model_order, ordered=True)
    df_plot["description"] = pd.Categorical(df_plot["description"], categories=desc_order, ordered=True)
    # task | description
    df_plot["task"] = pd.Categorical(df_plot["task"], categories=["XOR", "XNOR"], ordered=True)
    df_plot["task | description"] = df_plot["task"].astype(str) + " | " + df_plot["description"].astype(str)
    df_plot = df_plot.sort_values(["model_name", "task", "description"])  # stable alignment for error bars

    plt.figure(figsize=(14, 4))
    # ax = sns.barplot(data=df_plot, x="model_name", y="accuracy", hue="task", palette=["#377eb8", "#ff7f00"])
    # use "task | description" hue with shaded colors for different descriptions
    COLS = {
        "XOR | Only Examples": "#6b92b3",
        "XNOR | Only Examples": "#f8c18a",
        "XOR | Examples + Text Description": "#3d80b7",
        "XNOR | Examples + Text Description": "#fba34a",
        "XOR | Examples + Keyword": "#0265b6",
        "XNOR | Examples + Keyword": "#ff7f00",
    }
    ax = sns.barplot(
        data=df_plot,
        x="model_name",
        y="accuracy",
        hue="task | description",
        palette=COLS,
    )

    # Overlay per-description bars with hatches by drawing transparent bars on top.
    hatch_by_desc = {
        "Only Examples": "",
        "Examples + Text Description": ".",
        "Examples + Keyword": "/",
    }
    bars = ax.patches
    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        bar.set_hatch(hatch_by_desc[str(row["description"])])
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.errorbar(x=x, y=y, yerr=row["accuracy_se"], fmt="none", c="black", capsize=2)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticklabels([m.split("/")[-1] for m in model_order], rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "xor_xnor_multimodel_prompt_description_accuracy.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "xor_xnor_multimodel_prompt_description_accuracy.png"),
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XOR/XNOR keyword motivation experiment.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-7B",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Ministral-3-8B-Base-2512",
            "microsoft/phi-4",
        ],
    )
    parser.add_argument("--bit-length", type=int, default=3)
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "figs"))
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "exports", "xor_exp", "xor_xnor_multimodel_results.csv"),
    )
    args = parser.parse_args()

    df = run_experiment(
        model_names=args.models,
        bit_length=args.bit_length,
        num_shots=args.num_shots,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    df.to_csv(args.csv_path, index=False)
    print(f"Saved results CSV to {args.csv_path}")

    plot_results(df, output_dir=args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
