import random
from typing import Dict, Any, List

class Grader:
    def __init__(self):
        # Dummy blacklist for Easy task
        self.sanctioned_ids = ["ACC-BL-001", "ACC-BL-002", "ACC-BL-999"]
    
    def grade(self, tier: str, action: Any, current_target_id: str, current_transactions: List[Dict[str, Any]]) -> float:
        """Centralized grading entry point for all tiers."""
        is_flagged = action.action_type in ["FLAG", "BLOCK", "REQUEST_INFO"]
        p_id, p_cit = 0.0, 0.0
        f_p, f_n = 0.0, 0.0

        if tier == "easy":
            p_id = self.grade_easy(action.target_id, is_flagged)
            is_sanctioned = action.target_id in self.sanctioned_ids
            if not is_sanctioned and is_flagged: f_p = 1.0
            elif is_sanctioned and not is_flagged: f_n = 5.0
                
        elif tier == "medium":
            p_id = self.grade_medium(current_transactions, action.target_id, is_flagged)
            smurf_count = sum(1 for tx in current_transactions if 9000 <= tx.get("amount", 0) < 10000)
            has_smurfing = smurf_count >= 3
            if not has_smurfing and is_flagged: f_p = 1.0
            elif has_smurfing and not is_flagged: f_n = 5.0
                
        elif tier == "hard":
            p_id = 1.0 if is_flagged else 0.0
            p_cit = self.grade_hard(action.regulation_citation)
            if not is_flagged: f_n = 5.0

        # Reward = 0.5 * ID + 0.5 * Citation - 1.0 * FP - 1.0 * FN 
        raw_reward = (0.5 * p_id) + (0.5 * p_cit) - (1.0 * f_p) - (1.0 * f_n)
        
        # Proper normalization: squash unbounded raw rewards into strict (0, 1) range using sigmoid
        import math
        reward = 1.0 / (1.0 + math.exp(-raw_reward))
        return float(reward)

    def grade_easy(self, target_id: str, is_flagged: bool) -> float:
        """
        Sanction Hit: Binary match of AccountID vs Blacklist.
        Returns identification accuracy (Pid)
        """
        is_sanctioned = target_id in self.sanctioned_ids
        if is_sanctioned and is_flagged:
            return 1.0
        elif not is_sanctioned and not is_flagged:
            return 1.0
        return 0.0

    def grade_medium(self, transactions: List[Dict[str, Any]], target_id: str, is_flagged: bool) -> float:
        """
        Smurfing Detection: Temporal window analysis of related transactions.
        Checks for multiple transactions just under $10,000.
        """
        smurf_count = sum(1 for tx in transactions if 9000 <= tx.get("amount", 0) < 10000)
        has_smurfing = smurf_count >= 3
        
        if has_smurfing and is_flagged:
            return 1.0
        elif not has_smurfing and not is_flagged:
            return 1.0
        return 0.0

    def grade_hard(self, citation: str) -> float:
        """
        Regulatory Mapping: Case-insensitive check for mandatory regulations.
        """
        if not citation:
            return 0.0
            
        cit_upper = citation.upper()
        # Look for EU-AI-Act (Art 57) or BSA (31 USC 5318)
        valid_found = False
        if "EU-AI-ACT" in cit_upper and "57" in cit_upper:
            valid_found = True
        if "BSA" in cit_upper and "5318" in cit_upper:
            valid_found = True
        if "31-USC-5318" in cit_upper:
            valid_found = True

        if valid_found:
            return 1.0
        elif len(citation) > 15:
            # Partial credit if they cited something that looks like a regulation
            if any(term in cit_upper for term in ["ART", "SECTION", "USC", "REGULATION", "ACT"]):
                return 0.6
            return 0.3
        return 0.0
