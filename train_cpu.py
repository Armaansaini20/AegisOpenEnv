"""
AegisGym CPU-Optimized RL Training (REINFORCE)
Manual policy gradient loop for training without a GPU.
"""
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from client_env import get_sync_client
from server import AegisGymEnv
from models import AuditAction
from datasets import load_dataset
import itertools

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ENV_URL    = os.getenv("AEGISGYM_URL", "https://armaan020-aegisgym.hf.space")
LR         = 1e-5
EPISODES   = 5
ACCUMULATION_STEPS = 4

SYSTEM_PROMPT = """You are a financial compliance auditor AI.
Given a transaction scenario and regulatory context, respond with a JSON object:
{"action_type": "FLAG|APPROVE|REQUEST_INFO", "target_id": "<account_id>", "regulation_citation": "<regulation>"}
Be precise and concise."""

# ─── Dataset ──────────────────────────────────────────────────────────────────
print(f"Loading dataset: SecureFinAI-Lab/Regulations_QA...")
try:
    raw_dataset = load_dataset("SecureFinAI-Lab/Regulations_QA", split="train", streaming=True)
    dataset_iterator = itertools.cycle(iter(raw_dataset))
    print("Dataset loaded via streaming.")
except Exception as e:
    print(f"Dataset loading failed: {e}. Using generic prompts.")
    dataset_iterator = None

# ─── Model & Opt ─────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on CPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

# ─── Action Parser ────────────────────────────────────────────────────────────
def parse_action(text: str) -> AuditAction:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return AuditAction(**data)
    except Exception:
        pass
    return AuditAction(action_type="REQUEST_INFO", target_id="UNKNOWN", regulation_citation="parse_error")

# ─── Training Loop ────────────────────────────────────────────────────────────
def train():
    print(f"\n=== Starting CPU training (REINFORCE) ===")
    env = get_sync_client(ENV_URL)
    
    total_reward = 0
    
    for i in range(EPISODES):
        print(f"Episode {i+1}/{EPISODES}...")
        
        # Reset Env
        obs_payload = env.reset()
        obs = obs_payload.get("observation", {})
        
        # Build Prompt (with dataset augmentation)
        dataset_context = ""
        custom_prompt = "Audit the following transaction."
        if dataset_iterator:
            item = next(dataset_iterator)
            custom_prompt = item.get("question", custom_prompt)
            dataset_context = f"\nRegulatory Context: {item.get('answer', '')}"
            
        tier = env.state().get("current_tier", "easy")
        user_msg = (
            f"{custom_prompt}\n\n"
            f"Tier: {tier.upper()}\n"
            f"Transactions: {obs.get('transactions', [])}\n"
            f"Context: {obs.get('retrieved_regs', '')} {dataset_context}\n"
            f"Account: {obs.get('account_metadata', {})}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize and Generate with Gradients
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # We need to compute log-probs of the generated completion
        # and use those for the REINFORCE loss.
        # This is high-memory for long sequences, so we keep max_new_tokens small.
        outputs = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
        
        # Re-run full forward pass on the generated completion to get log_probs with gradients
        # The completion starts after inputs.input_ids.shape[1]
        full_ids = outputs.sequences
        context_len = inputs.input_ids.shape[1]
        completion_ids = full_ids[:, context_len:]
        
        logits = model(full_ids).logits
        # Align logits with completion ids (logits[i] is logit for token full_ids[i+1])
        completion_logits = logits[:, context_len-1:-1, :] 
        
        log_probs = F.log_softmax(completion_logits, dim=-1)
        # Gather the log-probs of the actual tokens chosen
        selected_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
        # Sum log-probs for the sequence (per REINFORCE)
        episode_log_prob = selected_log_probs.sum()
        
        # Execution
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        action = parse_action(completion_text)
        
        # Environment Step
        result = env.step(action.model_dump())
        reward = float(result.get("reward", 0.0))
        
        print(f"  Actions: {action.action_type} | Reward: {reward}")
        total_reward += reward
        
        # Loss: -Sum(log_probs) * Reward
        # Minimize this to maximize (log_probs * Reward)
        loss = -episode_log_prob * reward
        loss = loss / ACCUMULATION_STEPS # gradient accumulation
        
        loss.backward()
        
        if (i+1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("  --- Weights Updated! ---")
            
    print(f"\nTraining Complete. Avg Reward: {total_reward/EPISODES}")
    model.save_pretrained("aegisgym-cpu-agent")
    tokenizer.save_pretrained("aegisgym-cpu-agent")
    print(f"Model saved to 'aegisgym-cpu-agent'")

if __name__ == "__main__":
    train()
