"""
AegisGym GRPO Training Script
Connects to the live HF Space for environment rollouts.
Run: python train.py
"""
import os
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from client_env import get_sync_client
from models import AuditAction

# ─── Config ──────────────────────────────────────────────────────────────────
ENV_URL    = os.getenv("AEGISGYM_URL", "https://armaan020-aegisgym.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = "aegisgym-grpo-agent"

SYSTEM_PROMPT = """You are a financial compliance auditor AI.
Given a transaction scenario and regulatory context, respond with a JSON object:
{"action_type": "FLAG|APPROVE|REQUEST_INFO", "target_id": "<account_id>", "regulation_citation": "<regulation>"}
Be precise and concise."""

# ─── Action parser ────────────────────────────────────────────────────────────
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

# ─── Dataset Loading ──────────────────────────────────────────────────────────
from datasets import load_dataset
import itertools

print("Loading SecureFinAI-Lab/Regulations_QA dataset...")
try:
    # Use a streaming dataset for efficiency
    raw_dataset = load_dataset("SecureFinAI-Lab/Regulations_QA", split="train", streaming=True)
    dataset_iterator = itertools.cycle(iter(raw_dataset))
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load dataset: {e}. Falling back to default prompts.")
    dataset_iterator = None

# ─── Rollout function ─────────────────────────────────────────────────────────
def rollout_func(trainer, prompts, tokenizer):
    """One rollout episode connecting to the live AegisGym Space with dataset augmentation."""
    from trl.experimental.openenv import generate_rollout_completions
    
    env = get_sync_client(ENV_URL)
    result = env.reset()
    obs_dict = result.get("observation", {})
    
    # Sample from dataset if available
    dataset_context = ""
    custom_prompt = "Audit the following transaction."
    if dataset_iterator:
        item = next(dataset_iterator)
        custom_prompt = item.get("question", custom_prompt)
        dataset_context = f"\nRegulatory Context: {item.get('answer', '')}"
    
    all_prompt_ids, all_completion_ids, all_logprobs, rewards = [], [], [], []
    
    state = env.state()
    tier = state.get("current_tier", "easy")
    
    # Combine dataset context with environment context
    regs = obs_dict.get('retrieved_regs', "")
    retrieved_regs = [regs] if isinstance(regs, str) else list(regs)
    if dataset_context:
        retrieved_regs.append(dataset_context)
        
    user_msg = (
        f"{custom_prompt}\n\n"
        f"Tier: {tier.upper()}\n"
        f"Transactions: {obs_dict.get('transactions', [])}\n"
        f"Context: {retrieved_regs}\n"
        f"Account: {obs_dict.get('account_metadata', {})}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    out = generate_rollout_completions(trainer, [prompt_text])[0]
    all_prompt_ids.extend(out["prompt_ids"])
    all_completion_ids.extend(out["completion_ids"])
    all_logprobs.extend(out["logprobs"])
    
    completion_text = out.get("text") or tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
    action = parse_action(completion_text)
    result = env.step(action.model_dump())
    rewards.append(float(result.get("reward", 0.0)))
    
    return {
        "prompt_ids":      all_prompt_ids,
        "completion_ids":  all_completion_ids,
        "logprobs":        all_logprobs,
        "env_reward":      rewards[-1] if rewards else 0.0,
    }

# ─── Reward shim ─────────────────────────────────────────────────────────────
def reward_compliance(completions, **kwargs):
    rewards = kwargs.get("env_reward", [])
    if not rewards:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Model : {MODEL_NAME}")
    print(f"Env   : {ENV_URL}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = Dataset.from_dict({"prompt": ["Audit the following transaction."] * 200})
    
    config = GRPOConfig(
        num_train_epochs              = 1,
        learning_rate                 = 5e-6,
        per_device_train_batch_size   = 1,
        gradient_accumulation_steps   = 4,
        warmup_steps                  = 10,
        num_generations               = 2,
        max_completion_length         = 256,
        use_vllm                      = False,
        output_dir                    = OUTPUT_DIR,
        logging_steps                 = 1,
        save_steps                    = 25,
        gradient_checkpointing        = True,
    )
    
    trainer = GRPOTrainer(
        model            = MODEL_NAME,
        processing_class = tokenizer,
        reward_funcs     = [reward_compliance],
        train_dataset    = dataset,
        args             = config,
        rollout_func     = rollout_func,
    )
    
    print("\n=== Starting GRPO Training ===")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
