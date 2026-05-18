# Skill Neologisms

Official code for the paper: **[Skill Neologisms: Towards Skill-based Continual Learning](https://arxiv.org/abs/2605.04970)**, ICML 2026 (Spotlight).

## Setup

```bash
python3.10 -m venv skill_neologisms 
source skill_neologisms/bin/activate
# or with conda:
# conda create -n skill_neologisms python=3.10
# conda activate skill_neologisms
pip install -r requirements.txt
```

Environment variables:
- `WANDB_ENTITY`: required for all scripts that read/write WandB runs.
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_GPT5`: required for SkillMix GPT generation/grading (see `src/models/api_models.py`).

---

## Section 4: Motivating XOR Experiment

The following command reproduces the XOR/XNOR motivating experiment (Section 4):
```bash
PYTHONPATH=. python scripts/xor_experiment/xor_exp.py
```

---

## Section 5: Digit-Sequence Transformation 

*Note: We provide the fine-tuned Qwen2.5-0.5B model that we used as base model in our experiments directly from the [Hugging Face Hub](https://huggingface.co/antoninbrthn/skill-neologisms-qwen2.5-digitseq-base/blob/main/README.md). The configurations files are setup to automatically use this checkpoint. To pre-train the model from scratch, see the Appendix section below.*

### Train individual skills
To train a single skill neologism:
```bash
PYTHONPATH=. python sequence_map_experiment/train_neologisms.py \
  --config_name skill_tokens.yaml

# To override the skill operation and test operation:
PYTHONPATH=. python sequence_map_experiment/train_neologisms.py \
  --config_name skill_tokens.yaml \
  dataset.skill_op='"[SHIFT_RIGHT]"' dataset.test_op='"[ADD]"'
```

To train baselines models (LoRA / prompt tuning) on a single skill/test operation combination:
```bash
PYTHONPATH=. python sequence_map_experiment/train_baselines.py --config_name baseline_lora.yaml 

PYTHONPATH=. python sequence_map_experiment/train_prompt_tuning.py --config_name baseline_prompt_tuning.yaml
```

### Run all digit-sequence experiments

The following script trains all the methods (skill neologism, LoRA, and prompt-tuning) across all skill/test operations: 
```
bash sequence_map_experiment/run_digitseq_all.sh
```

### Evaluate skill composition vs ICL (Section 5.2.2)
After training neologisms for SHIFT and INVERT_POLARITY for different held-out test op, use this script to evaluate zero-shot neologism composition vs in-context learning (Section 5.2.2):
```bash
PYTHONPATH=. python sequence_map_experiment/evaluate_zs_compo_icl.py
```

### Build figures based on wandb logged runs
To reproduce Figures 4 and 5 from the paper based on the WandB runs from the experiments above:
```bash
PYTHONPATH=. python sequence_map_experiment/plot_results.py --targets p2_baselines  # Figure 4
PYTHONPATH=. python sequence_map_experiment/plot_results.py --targets p3_skill_vs_icl  # Figure 5
```

---

## Section 6: SkillMix Experiment

### Quickstart Demo: Skill Neologisms for LLaMA 3.2 3B

We provide skill-tokens for *Modus Ponens* and *Statistical Syllogism* hosted on the [Hugging Face Hub](https://huggingface.co/collections/antoninbrthn/skill-neologisms). 
The script below is a quick demo that automatically downloads these neologisms and compares LLaMA's output with and without them:

```bash
PYTHONPATH=. python scripts/skill-mix/demo_skill_mix.py
```

---

### Train skill neologisms models

By default, the skill neologism configs (eg `configs/skill_mix/skills/stat_modus.yaml`) use the trained skill tokens from the Hugging Face hub. The commands below show how to train skill neologisms from scratch. 

We provide the GPT-5 generated datasets used in the paper (`train_data_modus.csv` and `train_data_stat.csv`) under `data/skill_mix/`. To train skill neologisms using each dataset, use the following commands:
```bash
PYTHONPATH=. python scripts/skill-mix/train_skill_token.py -cn=train_skill_mix_stat
PYTHONPATH=. python scripts/skill-mix/train_skill_token.py -cn=train_skill_mix_modus
```

To re-create the training data via Azure OpenAI models rather than using the provided static files:
```bash
all_ks=(1 2 3)
for k in "${all_ks[@]}"; do
  PYTHONPATH=. python scripts/skill-mix/eval_skill_mix.py -cn=make_train_data_gpt5_stat skill_mix.k=$k
  PYTHONPATH=. python scripts/skill-mix/eval_skill_mix.py -cn=make_train_data_gpt5_modus skill_mix.k=$k 
done
# build datasets from generated data
PYTHONPATH=. python scripts/skill-mix/make_dataset.py --tag train_data_stat
PYTHONPATH=. python scripts/skill-mix/make_dataset.py --tag train_data_modus
# then update the dataset configs `configs/skill_mix/dataset/main/stat_{syllogism,modus}.yaml` to point to the generated csv files  
```


### Train baselines
To train LoRA and prompt-tuning baselines on the same datasets:
```bash
# LoRA
PYTHONPATH=. python scripts/skill-mix/train_baselines.py -cn=train_baseline_lora dataset=main/stat_syllogism
PYTHONPATH=. python scripts/skill-mix/train_baselines.py -cn=train_baseline_lora dataset=main/modus

# Prompt tuning
PYTHONPATH=. python scripts/skill-mix/train_prompt_tuning.py -cn=train_baseline_pt dataset=main/stat_syllogism
PYTHONPATH=. python scripts/skill-mix/train_prompt_tuning.py -cn=train_baseline_pt dataset=main/modus
```

### Evaluate trained skill neologisms and baselines on composition of both target skills

```bash
# eval with only one of the two skill neologisms
PYTHONPATH=. python scripts/skill-mix/eval_skill_mix.py -cn=eval_neologisms_combine_two skills=stat
PYTHONPATH=. python scripts/skill-mix/eval_skill_mix.py -cn=eval_neologisms_combine_two skills=modus
# eval with both skills neologisms
PYTHONPATH=. python scripts/skill-mix/eval_skill_mix.py -cn=eval_neologisms_combine_two skills=stat_modus

# eval baselines trained with either skills
PYTHONPATH=. python scripts/skill-mix/eval_baselines.py --config_name eval_baseline_combine_two.yaml --k 2 --splits ood
```

Plot results:
```bash
PYTHONPATH=. python scripts/skill-mix/plot_skill_mix.py 
```

---

## Appendix

### A. Pre-train the base digit-sequence model from scratch

To train a digit-sequence base model from scratch (default: Qwen2.5-0.5B):
```bash
PYTHONPATH=. python sequence_map_experiment/pretraining_qwen.py qwen.yaml
```

The final checkpoint will be saved under `sequence_map_experiment/checkpoints/run{i}/phase-2/final_model`. Then, manually set `pretrained: checkpoint: "run{i}/phase-2/final_model"` in the YAML configs inside `sequence_map_experiment/configs/` before training baselines or neologisms.


# Citation

```
@article{berthon2026skill,
  title={Skill Neologisms: Towards Skill-based Continual Learning},
  author={Berthon, Antonin and Astorga, Nicolas and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2605.04970},
  year={2026}
}
```