from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import List, Dict, Any

class AuditAction(Action):
    action_type: str = Field(description="One of: ['APPROVE', 'FLAG', 'REQUEST_INFO']")
    target_id: str = Field(description="The ID of the account or transaction being evaluated.")
    regulation_citation: str = Field(description="The specific regulation clause cited.")

class AuditObservation(Observation):
    transactions: List[Dict[str, Any]] = Field(description="List of transaction dicts")
    account_metadata: Dict[str, Any] = Field(description="Account metadata")
    retrieved_regs: str = Field(description="RAG-retrieved sections of guidelines")
    
    # Official OpenEnv returns these properties directly on the Observation
    reward: float = Field(default=0.0, description="Reward gained in the step")
    done: bool = Field(default=False, description="Whether the episode is complete")

class AuditState(State):
    step_count: int = Field(description="Current step of the episode")
    current_tier: str = Field(description="The compliance tier active for the episode")
    
    # Required by base State (depending on library version, might require extra fields, but standard is dict-like or these fields)
