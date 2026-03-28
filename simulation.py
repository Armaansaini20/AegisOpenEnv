"""
AegisGym Simulation Script
Runs multiple audit episodes using the LLM for inference (CPU-friendly).
This generates the logs and metrics to analyze the system's performance.
"""
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from client_env import get_sync_client
from train import parse_action, SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_URL = "https://armaan020-aegisgym.hf.space"

def run_simulation(num_episodes=5):
    print(f"=== Starting AegisGym Simulation (Inference Only) ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Env:   {ENV_URL}\n")

    print(f"Loading model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")
    print("Model loaded.\n")

    env = get_sync_client(ENV_URL)
    
    total_reward = 0
    results = []

    for i in range(num_episodes):
        print(f"--- Episode {i+1} ---")
        result = env.reset()
        obs_dict = result.get("observation", {})
        
        state = env.state()
        tier = state.get("current_tier", "easy")
        
        user_msg = (
            f"Audit the following transaction.\n\n"
            f"Tier: {tier.upper()}\n"
            f"Transactions: {obs_dict.get('transactions', [])}\n"
            f"Context: {obs_dict.get('retrieved_regs', [])}\n"
            f"Account: {obs_dict.get('account_metadata', {})}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        print(f"[Audit Prompt]:\n{user_msg}")
        
        # Inference
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[Model Reasoning]:\n{completion}")
        
        action = parse_action(completion)
        print(f"[Action]: {action.action_type} on {action.target_id}")
        
        step_result = env.step(action.model_dump())
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        
        print(f"[Reward]: {reward} | [Done]: {done}\n")
        total_reward += reward
        results.append({
            "episode": i+1,
            "tier": tier,
            "action": action.action_type,
            "reward": reward
        })

    print(f"=== Simulation Complete ===")
    print(f"Average Reward: {total_reward / num_episodes}")

if __name__ == "__main__":
    run_simulation(num_episodes=3)
