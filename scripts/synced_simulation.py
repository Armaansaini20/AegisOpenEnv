"""
AegisGym Synchronized Simulation
Saves exact dataset entries alongside the audit logs they generated.
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from client_env import get_sync_client
from train import parse_action, SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_URL = "https://armaan020-aegisgym.hf.space"

def run_synced_simulation(num_episodes=3):
    print(f"Loading model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")
    
    print("Loading dataset...")
    ds = load_dataset("SecureFinAI-Lab/Regulations_QA", split="train", streaming=True)
    it = iter(ds)
    
    env = get_sync_client(ENV_URL)
    full_report = []

    for i in range(num_episodes):
        print(f"--- Episode {i+1} ---")
        item = next(it)
        result = env.reset()
        obs_dict = result.get("observation", {})
        state = env.state()
        tier = state.get("current_tier", "easy")
        
        custom_prompt = item.get("question", "Audit the following transaction.")
        dataset_answer = item.get("answer", "No specific guidance provided.")
        
        user_msg = (
            f"{custom_prompt}\n\n"
            f"Tier: {tier.upper()}\n"
            f"Transactions: {obs_dict.get('transactions', [])}\n"
            f"Context: {obs_dict.get('retrieved_regs', [])}\n"
            f"Regulatory Hint: {dataset_answer}\n"
            f"Account: {obs_dict.get('account_metadata', {})}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        action = parse_action(completion)
        step_result = env.step(action.model_dump())
        
        full_report.append({
            "episode": i+1,
            "dataset_question": custom_prompt,
            "dataset_answer": dataset_answer,
            "tier": tier,
            "llm_reasoning": completion,
            "action": action.model_dump(),
            "reward": step_result.get("reward", 0.0)
        })

    with open("synced_report.json", "w") as f:
        json.dump(full_report, f, indent=2)
    print("\nSaved synced_report.json")

if __name__ == "__main__":
    run_synced_simulation()
