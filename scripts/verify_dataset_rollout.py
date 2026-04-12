"""
Verification script for dataset-augmented rollout function.
"""
from transformers import AutoTokenizer
from train import rollout_func, dataset_iterator
import trl.experimental.openenv

def verify():
    print("=== Verifying Dataset-Augmented Rollout ===")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Mock generation
    def mock_gen(trainer, prompts):
        print(f"\n[Generated Prompt Sample]:\n{prompts[0][:500]}...\n")
        fake_completion = '{"action_type": "APPROVE", "target_id": "ACC-001", "regulation_citation": "EU-Regulation"}'
        return [{"prompt_ids": [1], "completion_ids": [1], "logprobs": [0.0], "text": fake_completion}]
    
    trl.experimental.openenv.generate_rollout_completions = mock_gen
    
    print("Running rollout_func...")
    result = rollout_func(None, ["prompt"], tokenizer)
    print(f"\nRollout Finished!")
    print(f"Reward: {result['env_reward']}")

if __name__ == "__main__":
    verify()
