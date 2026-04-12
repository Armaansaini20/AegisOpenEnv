---
title: ComplianceEnv
emoji: 🏦
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
license: mit
tags:
  - openenv
---
# Compliance Environment (AegisGym)

This repository implements the **ComplianceEnv** sandbox, a financial auditing simulation for the Meta OpenEnv competition. It evaluates autonomous agents on their ability to identify sanctioned entities, detect smurfing patterns, and cite regulatory clauses.

## 🚀 Getting Started

### 1. Server Setup (Docker)

The environment is designed to run in a containerized environment (matching Hugging Face Spaces).

```bash
docker build -t aegis-gym:latest ./server
docker run -p 7860:7860 aegis-gym:latest
```

The server exposes standard OpenEnv endpoints:
- `GET /health`: System health check.
- `GET /metadata`: Environment metadata (name, description, version).
- `GET /schema`: Action and Observation schemas.
- `POST /reset`: Initialize a new audit episode.
- `POST /step`: Submit an audit action and receive reward/observation.
- `GET /state`: Inspect internal environment state.

### 2. Manual Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r server/requirements.txt
python -m server.app
```

### 3. Running Simulations

You can run a local simulation using the provided evaluation script:

```bash
python scripts/simulation.py
```

## 🏗️ Project Structure

- `server/`: Core environment implementation.
  - `app.py`: FastAPI entry point.
  - `server.py`: ComplianceEnv logic (stateless persistence).
  - `grader.py`: Tier-based reward calculation.
  - `models.py`: Pydantic schemas for actions and observations.
- `scripts/`: Utilities for training, evaluation, and deployment.
- `client.py`: Reference HTTP client for interacting with the environment.
- `openenv.yaml`: OpenEnv runtime specification.

## ⚖️ Environment Specification

### Tiers & Tasks

| Difficulty | Name | Evaluation Focus |
| :--- | :--- | :--- |
| 🟢 **Easy** | Sanction Check | Deterministic matching of Account IDs against a blacklist. |
| 🟡 **Medium** | Smurfing Detection | Pattern recognition within temporal transaction windows. |
| 🔴 **Hard** | Regulatory Alignment | Legal reasoning and precise citation of EU-AI-Act or BSA clauses. |

### Reward Function

Rewards are calculated based on identification accuracy ($P_{id}$) and citation accuracy ($P_{cit}$), with penalties for False Positives ($F_p$) and False Negatives ($F_n$):

$$Reward = \text{Sigmoid}(0.5 P_{id} + 0.5 P_{cit} - F_p - F_n)$$

This produces a strict $(0, 1)$ range suitable for automated evaluation.

## 📜 Compliance

AegisGym is fully compliant with the **OpenEnv 1.0.0** specification.
- **Statelessness**: State is persisted per request via file-based storage.
- **MCP Support**: Full Model Context Protocol integration via `/mcp`.
- **Validation**: Passed all deep-validation criteria for Meta OpenEnv Phase 2.
