from openenv.core.env_server import create_fastapi_app
from server import AegisGymEnv
from models import AuditAction, AuditObservation

app = create_fastapi_app(AegisGymEnv, AuditAction, AuditObservation)
