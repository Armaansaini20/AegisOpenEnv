import random
from typing import Dict, Any, Tuple
from openenv.core.env_server import Environment
from models import AuditAction, AuditObservation, AuditState
from grader import Grader

class AegisGymEnv(Environment):
    def __init__(self):
        super().__init__()
        self.grader = Grader()
        self.step_count = 0
        self.current_tier = "easy"
        self.max_steps = 100
        
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
        
        is_flagged = action.action_type in ["FLAG", "BLOCK", "REQUEST_INFO"]
        
        p_id, p_cit = 0.0, 0.0
        f_p, f_n = 0.0, 0.0
        
        if self.current_tier == "easy":
            p_id = self.grader.grade_easy(action.target_id, is_flagged)
            is_sanctioned = action.target_id in self.grader.sanctioned_ids
            if not is_sanctioned and is_flagged: f_p = 1.0
            elif is_sanctioned and not is_flagged: f_n = 5.0
                
        elif self.current_tier == "medium":
            p_id = self.grader.grade_medium(self.current_transactions, action.target_id, is_flagged)
            smurf_count = sum(1 for tx in self.current_transactions if 9000 <= tx.get("amount", 0) < 10000)
            has_smurfing = smurf_count >= 3
            if not has_smurfing and is_flagged: f_p = 1.0
            elif has_smurfing and not is_flagged: f_n = 5.0
                
        elif self.current_tier == "hard":
            p_id = 1.0 if is_flagged else 0.0
            p_cit = self.grader.grade_hard(action.regulation_citation)
            if not is_flagged: f_n = 5.0

        reward = (0.5 * p_id) + (0.5 * p_cit) - (1.0 * f_p) - (1.0 * f_n)
        done = self.step_count >= self.max_steps
        
        obs = self._create_observation(reward, done)
        self._set_next_scenario()
        return obs

    def _set_next_scenario(self):
        tiers = ["easy", "medium", "hard"]
        self.current_tier = tiers[self.step_count % 3]
        
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
