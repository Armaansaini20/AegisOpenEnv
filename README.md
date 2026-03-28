# AegisGym: Financial Compliance & AML Sandbox

AegisGym is a reinforcement learning sandbox for training and evaluating autonomous financial auditors. It simulates real-world banking compliance tasks, including Sanction Checks, Anti-Money Laundering (AML) detection, and Regulatory Alignment.

## 🏦 Motivation
Financial institutions process millions of transactions daily. Human auditors often struggle with "smurfing" hidden in noise or complex regulatory clauses across jurisdictions. AegisGym provides a rigorous environment to train LLM-based agents to detect financial crime with high explainability and logical reasoning.

## 🛠️ Environment Specification

### 📝 Action Space
The agent must provide an `AuditAction`:
- `action_type`: `APPROVE`, `FLAG`, or `REQUEST_INFO`.
- `target_id`: The ID of the account or transaction.
- `regulation_citation`: A string citing the relevant regulation (e.g., "BSA-31-USC-5318").

### 👁️ Observation Space
The agent receives an `AuditObservation`:
- `transactions`: A list of recent transaction dictionaries.
- `account_metadata`: Details about the account age, tier, and history.
- `retrieved_regs`: Relevant regulatory guidelines fetched via RAG.

### 🎯 Tasks & Difficulty
| Task ID | Name | Difficulty | Description |
|---------|------|------------|-------------|
| `easy_audit` | Sanction Check | Easy | Identify accounts on a blocklist. |
| `medium_audit` | Smurfing Detection | Medium | Detect structuring patterns under withdrawal limits. |
| `hard_audit` | Regulatory Alignment | Hard | Accurately cite complex regulations for high-risk tx. |

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- `openenv-core`
- Hugging Face Space (for deployment)

### Installation
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Baseline Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your_key"
python inference.py
```

## 🐋 Deployment
This environment is designed for Hugging Face Spaces. Use the provided `Dockerfile`.
- **Public URL:** [armaan020/AegisGym](https://huggingface.co/spaces/armaan020/AegisGym)
