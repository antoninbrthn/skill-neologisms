#!/usr/bin/env bash
set -euo pipefail

# Train skill neologisms, LoRA, and prompt tuning on the digit-sequence task (two new skills, leave-one-out train sets).

num_samples=100000  # full run 
# num_samples=1000  # uncomment for test run 
skill_ops=("[SHIFT_RIGHT]" "[INVERT_POLARITY]")
test_ops=("[ADD]" "[SUB]" "[ASC]" "[DESC]" "[ID]" "[POLARITY]")

# Train skill neologisms
for skill_op in "${skill_ops[@]}"; do
  for test_op in "${test_ops[@]}"; do
    echo "Running skill neologisms for skill=${skill_op}, test=${test_op}"
    PYTHONPATH=. python sequence_map_experiment/train_neologisms.py \
      --config_name skill_tokens.yaml \
      dataset.skill_op="\"$skill_op\"" \
      dataset.test_op="\"$test_op\"" \
      dataset.num_samples=$num_samples
  done
done

# Train prompt tuning baseline
for skill_op in "${skill_ops[@]}"; do
  for test_op in "${test_ops[@]}"; do
    echo "Running prompt tuning baseline for skill=${skill_op}, test=${test_op}"
    PYTHONPATH=. python sequence_map_experiment/train_prompt_tuning.py \
      --config_name baseline_prompt_tuning.yaml \
      dataset.skill_op="\"$skill_op\"" \
      dataset.test_op="\"$test_op\"" \
      dataset.num_samples=$num_samples
  done
done

# Train LoRA baseline
for skill_op in "${skill_ops[@]}"; do
  for test_op in "${test_ops[@]}"; do
    echo "Running LoRA baseline for skill=${skill_op}, test=${test_op}"
    
    PYTHONPATH=. python sequence_map_experiment/train_baselines.py \
      --config_name baseline_lora.yaml \
      dataset.skill_op="\"$skill_op\"" \
      dataset.test_op="\"$test_op\"" \
      dataset.num_samples=$num_samples
  done
done
