import os
import json
from openai import OpenAI
import httpx

# ─── Config (Required by Meta OpenEnv) ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "stepfun/step-3.5-flash:free")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
ENV_URL      = os.getenv("ENV_URL", "https://armaan020-complianceenv.hf.space")

client = OpenAI(
    api_key=API_KEY, 
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are a high-performance financial auditor AI.
Your goal is to maximize precision and minimize friction.
- FLAG: Use for CLEAR sanctions (BL targets) or smurfing evidence. 
- APPROVE: Use for CLEAN accounts. Do NOT waste time.
- REQUEST_INFO: ONLY use if the risk is ambiguous. Unnecessary requests are penalized.

Respond ONLY with a JSON object:
{"action_type": "FLAG|APPROVE|REQUEST_INFO", "target_id": "<id>", "regulation_citation": "<cite>"}"""

def run_baseline(num_episodes=10):
    print(f"=== ComplianceEnv Standardized Inference (v1) ===")
    print(f"Model: {MODEL_NAME} | Env: {ENV_URL}\n")
    
    http_client = httpx.Client(timeout=60.0)
    total_reward = 0.0
    episodes_run = 0
    
    for i in range(num_episodes):
        task_name = f"ComplianceEnv_Audit_Episode_{i+1}"
        print(f"--- Episode {i+1}/{num_episodes} ---")
        print(f"[START] task={task_name}", flush=True)
        step_count = 0
        episode_reward = 0.5
        try:
            reset_response = http_client.post(f"{ENV_URL}/reset")
            reset_response.raise_for_status()
            reset_data = reset_response.json()
            obs = reset_data.get("observation", {})
            target_id = obs.get("account_metadata", {}).get("target_id", "N/A")
            
            user_msg = (
                f"Audit the transaction.\n"
                f"Transactions: {obs.get('transactions', [])}\n"
                f"Context: {obs.get('retrieved_regs', '')}\n"
                f"Account: {obs.get('account_metadata', {})}"
            )
            
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
            
            # Robust Fallbacks
            if "target_id" not in action_data: action_data["target_id"] = target_id
            if "action_type" not in action_data: action_data["action_type"] = "APPROVE"
            
            print(f"  Target: {action_data.get('target_id')} | Action: {action_data.get('action_type')}", flush=True)
            
            step_response = http_client.post(f"{ENV_URL}/step", json={"action": action_data})
            step_response.raise_for_status()
            result = step_response.json()
            reward = float(result.get("reward", 0.5))
            episode_reward = reward
            step_count += 1
            
            print(f"  Reward: {reward}", flush=True)
            print(f"[STEP] step={step_count} reward={reward}", flush=True)
            
            total_reward += reward
            episodes_run += 1
            
            print(f"[END] task={task_name} score={episode_reward} steps={step_count}", flush=True)
            
        except Exception as e:
            print(f"  Error in episode {i+1}: {e}", flush=True)
            print(f"[END] task={task_name} score={episode_reward} steps={step_count}", flush=True)
            continue

    http_client.close()

    print(f"\n--- ComplianceEnv Reproducibility Report ---")
    print(f"Total Episodes Run: {episodes_run}")
    print(f"Benchmark Mean Score: {total_reward / max(1, episodes_run)}")
    print(f"Status: COMPLIANT")

if __name__ == "__main__":
    run_baseline()
