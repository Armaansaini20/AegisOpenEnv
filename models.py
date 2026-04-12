"""
Data models for the Compliance Environment.

These models define the action and observation types used by the OpenEnv
integration for the compliance auditor server.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server import Action, Observation


class ComplianceAction(Action):
    """
    Action taken by the compliance auditor agent.

    action_type values:
    - "APPROVE": Clear the transaction as legitimate.
    - "FLAG": Flag the transaction for sanctions/smurfing violation.
    - "REQUEST_INFO": Request additional information before deciding.
    """

    action_type: str = Field(
        ..., description="Action to take: APPROVE, FLAG, or REQUEST_INFO"
    )
    target_id: str = Field(
        ..., description="Identifier of the account or transaction being audited"
    )
    regulation_citation: Optional[str] = Field(
        None,
        description="Direct citation from the retrieved regulations (Required for Hard tasks)",
    )


class ComplianceObservation(Observation):
    """Observation returned by the compliance environment after each step."""

    transactions: List[Dict[str, Any]] = Field(
        ..., description="Chronological list of recent transactions for the target entity"
    )
    account_metadata: Dict[str, Any] = Field(
        ..., description="Metadata about the account (age, risk level, country)"
    )
    retrieved_regs: str = Field(
        ..., description="Text-based regulatory context window providing relevant compliance clauses"
    )
    reward: float = Field(..., description="Immediate reward for the last taken action")
    done: bool = Field(..., description="Episode completion flag")


__all__ = [
    "ComplianceAction",
    "ComplianceObservation",
]
