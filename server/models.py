from typing import List, Dict, Any, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State

class AuditAction(Action):
    """Action taken by the auditor agent."""
    action_type: str = Field(..., description="Action to take: APPROVE, FLAG, or REQUEST_INFO")
    target_id: str = Field(..., description="Identifier of the account or transaction being audited")
    regulation_citation: Optional[str] = Field(None, description="Direct citation from the retrieved regulations (Required for Hard tasks)")

class AuditObservation(Observation):
    """Observation received by the agent after taking an action."""
    transactions: List[Dict[str, Any]] = Field(..., description="Chronological list of recent transactions for the target entity")
    account_metadata: Dict[str, Any] = Field(..., description="Metadata about the account (age, risk level, country)")
    retrieved_regs: str = Field(..., description="Text-based regulatory context window providing relevant compliance clauses")
    reward: float = Field(..., description="Immediate reward for the last taken action")
    done: bool = Field(..., description="Episode completion flag")

class AuditState(State):
    """Current internal state of the environment server."""
    step_count: int = Field(..., description="Current step in the audit episode")
    current_tier: str = Field(..., description="Difficulty tier: easy, medium, or hard")
