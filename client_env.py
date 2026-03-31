from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.sync_client import SyncEnvClient
from server.models import AuditAction, AuditObservation

HF_SPACE_WSS = "wss://armaan020-aegisopenenv.hf.space"

class AegisGymWsClient(EnvClient):
    """Concrete EnvClient implementation for the AegisOpenEnv HF Space."""

    def _step_payload(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an action dict into the WS step payload."""
        return action if isinstance(action, dict) else action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> Any:
        """Parse reset/step response from the server into usable result."""
        return payload

    def _parse_state(self, payload: Dict[str, Any]) -> Any:
        """Parse the state endpoint response."""
        return payload

def get_sync_client(ws_url: str = HF_SPACE_WSS) -> SyncEnvClient:
    """Return a synchronous wrapper over the WebSocket client."""
    return AegisGymWsClient(base_url=ws_url).sync()
