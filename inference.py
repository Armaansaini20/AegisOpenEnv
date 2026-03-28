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
HF_TOKEN     = os.getenv("HF_TOKEN", "") # Used if hitting HF Inference Endpoints
ENV_URL      = os.getenv("ENV_URL", "https://armaan020-aegisgym.hf.space")

client = OpenAI(api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY"), base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a financial compliance auditor AI.
Respond ONLY with a JSON object:
{"action_type": "FLAG|APPROVE|REQUEST_INFO", "target_id": "<id>", "regulation_citation": "<cite>"}"""

def run_baseline(num_steps=3):
    print(f"=== AegisGym Baseline Inference ===")
    print(f"Model: {MODEL_NAME} | Env: {ENV_URL}")
    
    env = get_sync_client(ENV_URL)
    
    for i in range(num_steps):
        print(f"\n--- Step {i+1} ---")
        obs_payload = env.reset()
        obs = obs_payload.get("observation", {})
        
        user_msg = (
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
        print(f"LLM Response: {content}")
        
        # Parse and Step
        try:
            action_data = json.loads(content)
            result = env.step(action_data)
            print(f"Reward: {result.get('reward')} | Done: {result.get('done')}")
        except Exception as e:
            print(f"Parse Error: {e}")

if __name__ == "__main__":
    run_baseline()
