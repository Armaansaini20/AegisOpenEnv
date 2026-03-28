import random
from typing import Dict, Any, List

class Grader:
    def __init__(self):
        # Dummy blacklist for Easy task
        self.sanctioned_ids = ["ACC-BL-001", "ACC-BL-002", "ACC-BL-999"]
    
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
        Regulatory Mapping: LLM-as-a-judge + Cosine similarity (mocked here).
        Returns citation accuracy (Pcit) based on simulated semantic match.
        """
        if not citation:
            return 0.0
            
        # Simplified mock evaluation
        valid_citations = ["EU-AI-Act-Art-57", "BSA-31-USC-5318"]
        if any(v in citation for v in valid_citations):
            return 1.0
        elif len(citation) > 10:
            return 0.5 # Partial credit for trying to cite something substantive
        return 0.0
