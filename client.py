from openenv.core.env_client import EnvClient
from models import ComplianceAction
from server import ComplianceEnv
import json

def run_test():
    print("Initializing test environment...")
    
    # In a real setup, client talks to a served environment via EnvClient
    # For local demonstration, we instantiate the Env directly
    env = ComplianceEnv()
    
    obs = env.reset()
    print("Initial Observation:")
    print(obs.model_dump_json(indent=2))
    
    # Simulate a bad action (approve a sanctioned entity)
    # Let's force it to be a sanctioned entity for testing
    env.db.current_entity = "EvilCorp"
    
    action1 = ComplianceAction(
        decision="APPROVE",
        reasoning="The transaction seems fine.",
        cited_regulation_id="NONE"
    )
    
    new_obs, reward, done, info = env.step(action1)
    print("\nAction 1: APPROVE EvilCorp")
    print(f"Reward: {reward} (Expect negative)")
    print(f"Trace info: {info}")
    
    # Simulate a good action
    env.db.current_entity = "EvilCorp"
    action2 = ComplianceAction(
        decision="BLOCK",
        reasoning="Entity is on the sanctions list. Blocking as per AML regulations.",
        cited_regulation_id="EU-AML-01"
    )
    
    new_obs, reward, done, info = env.step(action2)
    print("\nAction 2: BLOCK EvilCorp with good reasoning")
    print(f"Reward: {reward} (Expect positive)")
    print(f"Trace info: {info}")

if __name__ == "__main__":
    run_test()
