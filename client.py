"""
Compliance Environment HTTP client.

This client uses the OpenEnv HTTP endpoints exposed by the compliance server:
/reset, /step, and /state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import httpx

try:
    from .models import ComplianceAction, ComplianceObservation
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from models import ComplianceAction, ComplianceObservation

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "response_output"


class ComplianceEnv:
    """HTTP client for the Compliance environment."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "ComplianceEnv":
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _parse_step_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[ComplianceObservation]:
        obs_data = payload.get("observation", {})
        observation = ComplianceObservation(
            transactions=obs_data.get("transactions", []),
            account_metadata=obs_data.get("account_metadata", {}),
            retrieved_regs=obs_data.get("retrieved_regs", ""),
            reward=obs_data.get("reward", 0.0),
            done=obs_data.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def reset(self) -> StepResult[ComplianceObservation]:
        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/reset",
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def step(
        self,
        action: ComplianceAction | Dict[str, Any],
    ) -> StepResult[ComplianceObservation]:
        if isinstance(action, ComplianceAction):
            action_payload = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            action_payload = action
        else:
            raise TypeError("action must be ComplianceAction or dict")

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/step",
            json={"action": action_payload},
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def state(self) -> Dict[str, Any]:
        client = self._ensure_client()
        response = client.get(
            f"{self.base_url}/state",
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()


@dataclass
class ScenarioConfig:
    gym_environment_url: str
    system_prompt: str
    user_prompt: str
    llm_model: str
    llm_provider: str
    llm_api_key: str
    number_of_runs: int = 1
    temperature: float = 0.0
    max_tokens: int = 4096
    max_iterations: int = 20
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _load_scenario(path: str | Path) -> ScenarioConfig:
    """Load scenario configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioConfig(
        gym_environment_url=data.get("gym_environment_url", "http://localhost:7860"),
        system_prompt=data.get("system_prompt", ""),
        user_prompt=data.get("user_prompt", ""),
        llm_model=data.get("llm_model", ""),
        llm_provider=data.get("llm_provider", "openai"),
        llm_api_key=data.get("llm_api_key", ""),
        number_of_runs=data.get("number_of_runs", 1),
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens", 4096),
        max_iterations=data.get("max_iterations", 20),
    )


def run_scenario(config: ScenarioConfig) -> Dict[str, Any]:
    """Execute a scenario benchmark using the OpenAI client."""
    from openai import OpenAI

    llm_client = OpenAI(
        api_key=config.llm_api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url="https://api.openai.com/v1"
        if config.llm_provider == "openai"
        else os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    env = ComplianceEnv(base_url=config.gym_environment_url)
    results = []

    for run_number in range(1, config.number_of_runs + 1):
        start_time = datetime.now(timezone.utc)
        print(f"\n--- Run {run_number}/{config.number_of_runs} ---")

        result = env.reset()
        obs = result.observation

        user_msg = (
            f"Audit the transaction.\n"
            f"Transactions: {obs.transactions}\n"
            f"Context: {obs.retrieved_regs}\n"
            f"Account: {obs.account_metadata}"
        )

        response = llm_client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        content = response.choices[0].message.content
        action_data = json.loads(content)

        target_id = obs.account_metadata.get("target_id", "N/A")
        if "target_id" not in action_data:
            action_data["target_id"] = target_id
        if "action_type" not in action_data:
            action_data["action_type"] = "APPROVE"

        step_result = env.step(action_data)
        reward = step_result.reward

        execution_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        run_result = {
            "run_number": run_number,
            "target_id": action_data.get("target_id"),
            "action_type": action_data.get("action_type"),
            "regulation_citation": action_data.get("regulation_citation"),
            "reward": reward,
            "execution_time_ms": execution_time_ms,
        }
        results.append(run_result)
        print(f"  Target: {run_result['target_id']} | Action: {run_result['action_type']} | Reward: {reward}")

    env.close()

    output = {
        "config": {
            "model": f"{config.llm_provider}/{config.llm_model}",
            "number_of_runs": config.number_of_runs,
        },
        "runs": results,
    }

    # Save output
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = config.output_dir / f"benchmark_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="ComplianceEnv Client")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path to scenario configuration JSON file",
    )
    args = parser.parse_args()

    config = _load_scenario(args.scenario)
    run_scenario(config)


if __name__ == "__main__":
    main()
