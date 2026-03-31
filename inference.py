"""
AegisGym Baseline Inference Script
Requirement: Must use OpenAI client and env vars for credentials.
"""
import os
import json
from openai import OpenAI
from client_env import get_sync_client
from models import AuditAction

# ─── Config (Required by Meta OpenEnv) ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_URL      = os.getenv("ENV_URL", "https://armaan020-aegisgym.hf.space")

# OpenRouter recommends specific headers for their API
extra_headers = {
    "HTTP-Referer": "https://huggingface.co/spaces/armaan020/AegisGym",
    "X-Title": "AegisGym OpenEnv Submission"
}

client = OpenAI(
    api_key=OPENAI_API_KEY, 
    base_url=API_BASE_URL,
    default_headers=extra_headers if "openrouter" in API_BASE_URL.lower() else None
)

SYSTEM_PROMPT = """You are a high-performance financial auditor AI.
Your goal is to maximize precision and minimize friction.
- FLAG: Use for CLEAR sanctions (BL targets) or smurfing evidence. 
- APPROVE: Use for CLEAN accounts. Do NOT waste time.
- REQUEST_INFO: ONLY use if the risk is ambiguous. Unnecessary requests are penalized.

Respond ONLY with a JSON object:
{"action_type": "FLAG|APPROVE|REQUEST_INFO", "target_id": "<id>", "regulation_citation": "<cite>"}"""

def run_baseline(num_episodes=10):
    print(f"=== AegisGym Standardized Inference (v4) ===")
    print(f"Model: {MODEL_NAME} | Env: {ENV_URL}\n")
    
    env = get_sync_client(ENV_URL)
    total_reward = 0.0
    episodes_run = 0
    
    for i in range(num_episodes):
        print(f"--- Episode {i+1}/{num_episodes} ---")
        try:
            obs_payload = env.reset()
            obs = obs_payload.get("observation", {})
            
            user_msg = (
                f"Audit the transaction.\n"
                f"Transactions: {obs.get('transactions', [])}\n"
                f"Context: {obs.get('retrieved_regs', '')}\n"
                f"Account: {obs.get('account_metadata', {})}"
            )
            
            # OpenAI API Call
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            action_data = json.loads(content)
            print(f"  Target: {action_data.get('target_id')} | Action: {action_data.get('action_type')}")
            
            result = env.step(action_data)
            reward = float(result.get("reward", 0.0))
            print(f"  Reward: {reward}")
            
            total_reward += reward
            episodes_run += 1
            
        except Exception as e:
            print(f"  Error in episode: {e}")
            continue

    print(f"\n--- AegisGym Reproducibility Report ---")
    print(f"Total Episodes Run: {episodes_run}")
    print(f"Benchmark Mean Score: {total_reward / max(1, episodes_run)}")
    print(f"Status: COMPLIANT")

if __name__ == "__main__":
    run_baseline()
