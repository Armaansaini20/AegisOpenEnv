import random
import json
import os
import threading
from typing import Dict, Any, List
from pathlib import Path
from openenv.core.env_server import Environment
from .models import ComplianceAction, ComplianceObservation, ComplianceState
from .grader import Grader


# Simple file-based state store
STATE_FILE = Path(os.getenv("STATE_DIR", "/tmp")) / "compliance_env_state.json"

def _load_state() -> Dict[str, Any]:
    """Load persisted environment state."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    """Persist environment state."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


class ComplianceEnv(Environment):
    """
    ComplianceEnv is a financial compliance sandbox that translates raw transactions
    into auditable tasks for LLM-based autonomous agents.

    It supports multiple tiers: Sanctions (Easy), Smurfing (Medium), and
    Regulatory (Hard) citation.

    NOTE: The OpenEnv HTTP framework creates a NEW instance for every request.
    State is persisted externally via a JSON file so that step() can grade
    based on the scenario established during reset().
    """

    current_tier: str = "easy"
    current_target_id: str = "ACC-BL-001"
    current_transactions: List[Dict[str, Any]] = []

    # Metadata for OpenEnv
    name: str = "compliance_env"
    description: str = "Financial compliance sandbox for auditing transactions across easy (Sanctions), medium (Smurfing), and hard (Regulatory) tiers."
    version: str = "1.0.0"

    def __init__(self):
        super().__init__()
        self.grader = Grader()

    def reset(self, **kwargs) -> ComplianceObservation:
        """Initialize a new episode and return the first observation."""
        # Pick a random tier and scenario
        tier = random.choice(["easy", "medium", "hard"])
        scenario = self._generate_scenario(tier)

        # Persist the state so step() can access it
        _save_state({
            "step_count": 0,
            "current_tier": tier,
            "current_target_id": scenario["target_id"],
            "current_transactions": scenario["transactions"],
            "max_steps": 10,
        })

        return ComplianceObservation(
            transactions=scenario["transactions"],
            account_metadata={
                "age_days": 120,
                "tier": "standard",
                "target_id": scenario["target_id"],
            },
            retrieved_regs="EU-AI-Act-Art-57: Sandboxes required for high risk. BSA-31-USC-5318: AML programs.",
            reward=0.0,
            done=False,
        )

    @property
    def state(self) -> ComplianceState:
        """Return the current environment state."""
        persisted = _load_state()
        return ComplianceState(
            step_count=persisted.get("step_count", 0),
            current_tier=persisted.get("current_tier", "easy"),
        )

    def step(self, action: ComplianceAction, **kwargs) -> ComplianceObservation:
        """Grade the agent's action against the current scenario."""
        persisted = _load_state()

        step_count = persisted.get("step_count", 0) + 1
        current_tier = persisted.get("current_tier", "easy")
        current_target_id = persisted.get("current_target_id", "ACC-BL-001")
        current_transactions = persisted.get("current_transactions", [])
        max_steps = persisted.get("max_steps", 10)

        # Grade the action based on the CURRENT scenario
        reward = self.grader.grade(
            tier=current_tier,
            action=action,
            current_target_id=current_target_id,
            current_transactions=current_transactions,
        )

        done = step_count >= max_steps

        # Advance to next scenario for the next step
        next_tier = self._next_tier(current_tier)
        next_scenario = self._generate_scenario(next_tier)

        # Save updated state
        _save_state({
            "step_count": step_count,
            "current_tier": next_tier,
            "current_target_id": next_scenario["target_id"],
            "current_transactions": next_scenario["transactions"],
            "max_steps": max_steps,
        })

        return ComplianceObservation(
            transactions=next_scenario["transactions"],
            account_metadata={
                "age_days": 120,
                "tier": "standard",
                "target_id": next_scenario["target_id"],
            },
            retrieved_regs="EU-AI-Act-Art-57: Sandboxes required for high risk. BSA-31-USC-5318: AML programs.",
            reward=reward,
            done=done,
        )

    def _next_tier(self, current: str) -> str:
        """Cycle through tiers: easy -> medium -> hard -> easy."""
        cycle = {"easy": "medium", "medium": "hard", "hard": "easy"}
        return cycle.get(current, "easy")

    def _generate_scenario(self, tier: str) -> Dict[str, Any]:
        """Generate a scenario for the given tier."""
        if tier == "easy":
            target = random.choice(["ACC-BL-001", "ACC-CLEAN-01"])
            transactions = [{"amount": 500, "currency": "USD"}]
        elif tier == "medium":
            target = "ACC-SMURF-99"
            if random.random() > 0.5:
                transactions = [{"amount": 9500, "currency": "USD"} for _ in range(4)]
            else:
                transactions = [
                    {"amount": 5000, "currency": "USD"},
                    {"amount": 200, "currency": "USD"},
                ]
        elif tier == "hard":
            target = "ACC-REG-VIOLATOR"
            transactions = [
                {
                    "amount": 50000,
                    "currency": "USD",
                    "note": "High risk AI deployment without sandbox",
                }
            ]
        else:
            target = "ACC-UNKNOWN"
            transactions = []

        return {"target_id": target, "transactions": transactions}
