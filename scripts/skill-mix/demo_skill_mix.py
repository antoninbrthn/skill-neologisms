import torch
import transformers
from src.models.loading import load_base_hf_model
from src.models.skill_token_model import SkillTokenModel, insert_skill_tokens
from src.skill_mix.custom_utils.data import (
    load_skills,
    build_generation_prompt,
)

transformers.set_seed(42)  # for reproducibility


def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Base model: {model_name}...")
    model, tokenizer, _ = load_base_hf_model(model_name)
    model.to(device)

    print("\nSetting up SkillToken adapter with skills from Hugging Face hub...")
    adapter = SkillTokenModel(model_name, device=device, model=model, tokenizer=tokenizer)
    adapter.load_skills("antoninbrthn/skill-neologisms-llama3-skillmix-modus", overwrite_existing=True)
    adapter.load_skills(
        "antoninbrthn/skill-neologisms-llama3-skillmix-stat-syllogism",
        overwrite_existing=True,
    )

    print("\nBuilding SkillMix prompt...")
    skills_dict = load_skills()
    skills_list = ["modus ponens", "statistical syllogism"]
    topic = "Gardening"
    instruction = build_generation_prompt(skills_list, topic, skills_dict)

    # Apply chat template
    messages = [{"role": "user", "content": instruction}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Insert the skill tokens where the skill strings appear
    out_prompts = [prompt_str]
    out_prompts = insert_skill_tokens(
        texts=out_prompts,
        skill_name="modus_ponens",
        replace_str="modus ponens",
        adapter=adapter,
    )
    out_prompts = insert_skill_tokens(
        texts=out_prompts,
        skill_name="stat_syllogism",
        replace_str="statistical syllogism",
        adapter=adapter,
    )
    prompt_with_tokens = out_prompts[0]

    def generate(p):
        inputs = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

    print("\n" + "=" * 50)
    print("DEMO: GENERATING WITHOUT SKILL NEOLOGISMS")
    print("=" * 50)
    print(f"Prompt text:\n{prompt_str}")
    print("-" * 50)
    print(f"Model Output:\n{generate(prompt_str)}\n")

    print("=" * 50)
    print("DEMO: GENERATING WITH SKILL NEOLOGISMS")
    print("=" * 50)
    print(f"Prompt text (tokens inserted):\n{prompt_with_tokens}")
    print("-" * 50)
    print(f"Model Output:\n{generate(prompt_with_tokens)}\n")


if __name__ == "__main__":
    main()
