import json

def generate_report():
    with open("synced_report.json", "r") as f:
        data = json.load(f)
        
    with open("matching_dataset_logs.md", "w", encoding="utf-8") as f:
        f.write("# Synchronized Audit & Dataset Report\n\n")
        f.write("This report pairs each simulation episode with the exact dataset entry that triggered it.\n\n")
        
        for e in data:
            f.write(f"## Episode {e['episode']}\n")
            f.write(f"### 📊 Dataset Entry\n")
            f.write(f"**Question:** {e['dataset_question']}\n\n")
            f.write(f"**Context (Answer):** {e['dataset_answer']}\n\n")
            f.write(f"### 📄 Audit Log\n")
            f.write(f"**LLM Reasoning:**\n```json\n{e['llm_reasoning']}\n```\n")
            f.write(f"**Action:** {e['action']['action_type']} on {e['action']['target_id']}\n\n")
            f.write(f"**Reward:** {e['reward']}\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    generate_report()
