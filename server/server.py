import random
from typing import Dict, Any, Tuple, List
from openenv.core.env_server import Environment
from .models import AuditAction, AuditObservation, AuditState
from .grader import Grader

class AegisOpenEnv(Environment):
    """
    AegisOpenEnv is a financial compliance sandbox that translates raw transactions
    into auditable tasks for LLM-based autonomous agents.
    
    It supports multiple tiers: Sanctions (Easy), Smurfing (Medium), and 
    Regulatory (Hard) citation.
    """
    def __init__(self):
        super().__init__()
        self.grader = Grader()
        self._history = []
        self.current_tier = "easy"
        self.step_count = 0
        self.max_steps = 10
        
        # Test Data
        self.current_target_id = "ACC-BL-001"
        self.current_transactions = []
        
    def reset(self) -> AuditObservation:
        self.step_count = 0
        self._set_next_scenario()
        return self._create_observation(0.0, False)

    @property
    def state(self) -> AuditState:
        return AuditState(
            step_count=self.step_count,
            current_tier=self.current_tier
        )
        
    def _create_observation(self, reward: float, done: bool) -> AuditObservation:
        return AuditObservation(
            transactions=self.current_transactions,
            account_metadata={"age_days": 120, "tier": "standard", "target_id": self.current_target_id},
            retrieved_regs="EU-AI-Act-Art-57: Sandboxes required for high risk. BSA-31-USC-5318: AML programs.",
            reward=reward,
            done=done
        )

    def step(self, action: AuditAction) -> AuditObservation:
        self.step_count += 1
        
        reward = self.grader.grade(
            tier=self.current_tier,
            action=action,
            current_target_id=self.current_target_id,
            current_transactions=self.current_transactions
        )
        
        done = self.step_count >= self.max_steps
        
        obs = self._create_observation(reward, done)
        self._set_next_scenario()
        return obs

    def _set_next_scenario(self):
        tiers = ["easy", "medium", "hard"]
        self.current_tier = tiers[self.step_count % len(tiers)]
        
        if self.current_tier == "easy":
            self.current_target_id = random.choice(["ACC-BL-001", "ACC-CLEAN-01"])
            self.current_transactions = [{"amount": 500, "currency": "USD"}]
        elif self.current_tier == "medium":
            self.current_target_id = "ACC-SMURF-99"
            if random.random() > 0.5:
                self.current_transactions = [{"amount": 9500, "currency": "USD"} for _ in range(4)]
            else:
                self.current_transactions = [{"amount": 5000, "currency": "USD"}, {"amount": 200, "currency": "USD"}]
        elif self.current_tier == "hard":
            self.current_target_id = "ACC-REG-VIOLATOR"
            self.current_transactions = [{"amount": 50000, "currency": "USD", "note": "High risk AI deployment without sandbox"}]
