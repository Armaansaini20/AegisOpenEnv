---
title: AegisOpenEnv
emoji: 🏦
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# 🏦 AegisOpenEnv: AI-Powered Financial Compliance Sandbox

**AegisOpenEnv** is a high-fidelity Reinforcement Learning environment designed for the **Meta OpenEnv** competition. It translates complex banking compliance regulations into a rigorous, text-augmented simulation for training autonomous financial auditors.

---

## 🏛️ Why AegisOpenEnv?
Financial institutions screen millions of transactions daily. Traditional rule-based systems often struggle with **"smurfing"** (structuring transactions just under reporting limits) or adapting to new **Sanctions Lists**. 

AegisOpenEnv allows LLM-based agents to:
- **Audit Raw Transactions**: Process complex histories and account metadata.
- **Reason with Regulations**: Dynamically fetch and cite clauses like the **EU AI Act** or **BSA**.
- **Learn from Feedback**: Use modular reward signals to optimize for high precision and low false positives.

---

## 🛠️ Task Catalog

Our environment features a 3-tier difficulty system to evaluate various auditor competencies:

| Phase | Task ID | Name | Difficulty | Competency Evaluated |
| :--- | :--- | :--- | :--- | :--- |
| **I** | `easy_audit` | Sanction Check | 🟢 Easy | Blacklist matching and deterministic identification. |
| **II** | `medium_audit` | Smurfing Detection | 🟡 Medium | Pattern recognition across temporal windows. |
| **III** | `hard_audit` | Regulatory Alignment | 🔴 Hard | Legal reasoning and precise clause citation. |

---

## 👁️ Environment Specification

### 📝 Action Space (`AuditAction`)
Agents respond with structured JSON containing:
- `action_type`: `APPROVE`, `FLAG`, or `REQUEST_INFO`.
- `target_id`: The identifier of the account or transaction under review.
- `regulation_citation`: A direct citation of the violated regulation (Required for Hard tier).

### 👁️ Observation Space (`AuditObservation`)
Agents receive:
- `transactions`: Real-time transaction flux.
- `account_metadata`: Profile data (age, tier, risk level).
- `retrieved_regs`: Dynamic context window containing regulatory guidelines.
- `reward`: The score from the previous action.

### 🎯 Reward Structure
AegisOpenEnv prioritizes **Zero-Tolerance Compliance**:
- **Successful Audit**: +0.5 to +1.0 (Identification + Citation).
- **False Positive**: -1.0 (Inefficiency penalty).
- **Missed Detection (False Negative)**: **-5.0** (Critical regulatory failure).

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Local Validation
```bash
# Start the server
uvicorn app:app --port 7860

# Run OpenEnv validate
openenv validate http://localhost:7860
```

### Inference Baseline
Ensure you have set your API credentials in your terminal session:
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
$env:API_BASE_URL = "https://openrouter.ai/api/v1"
$env:MODEL_NAME = "stepfun/step-3.5-flash:free"
$env:ENV_URL = "https://armaan020-aegisopenenv.hf.space"
python inference.py
```

---

## 🏁 Compliance Status
This environment is **100% Compliant** with the Meta OpenEnv specification.
- **Validation URL**: [armaan020-aegisopenenv.hf.space/health](https://armaan020-aegisopenenv.hf.space/health)
- **Repo Walkthrough**: View `walkthrough.md` for training logs and REINFORCE results.

---

