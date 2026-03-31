"""
AegisGym WebSocket client — concrete subclass of openenv EnvClient.

Usage:
    client = AegisGymWsClient()
    sync_client = client.sync()
    obs = sync_client.reset()
    obs = sync_client.step({...})
"""
from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.sync_client import SyncEnvClient
from server.models import AuditAction, AuditObservation

HF_SPACE_WSS = "wss://armaan020-aegisgym.hf.space"


class AegisGymWsClient(EnvClient):
    """Concrete EnvClient implementation for the AegisGym HF Space."""

    def _step_payload(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an action dict into the WS step payload."""
        return action if isinstance(action, dict) else action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> Any:
        """Parse reset/step response from the server into usable result."""
        return payload  # keep as dict; training code accesses .observation, .reward, .done

    def _parse_state(self, payload: Dict[str, Any]) -> Any:
        """Parse the state endpoint response."""
        return payload


def get_sync_client(ws_url: str = HF_SPACE_WSS) -> SyncEnvClient:
    """Return a synchronous wrapper over the WebSocket client."""
    return AegisGymWsClient(base_url=ws_url).sync()


def test_live_env():
    print("=== Testing Live AegisGym Space (wss) ===")
    client = get_sync_client()

    print("reset() ...")
    result = client.reset()
    print(f"  result type: {type(result)}")
    print(f"  keys: {list(result.keys()) if isinstance(result, dict) else dir(result)}")

    action = AuditAction(
        action_type="FLAG",
        target_id="ACC-BL-001",
        regulation_citation="EU-AI-Act-Art-57"
    ).model_dump()

    print("step(FLAG ACC-BL-001) ...")
    result = client.step(action)
    print(f"  reward={result.get('reward') if isinstance(result, dict) else getattr(result,'reward',None)}")
    print(f"  done={result.get('done') if isinstance(result, dict) else getattr(result,'done',None)}")

    print("state() ...")
    s = client.state()
    print(f"  State: {s}")

    print("\n=== Live environment OK! ===")


if __name__ == "__main__":
    test_live_env()
