from .server import AegisOpenEnv
from openenv.core.env_server import create_fastapi_app
from .models import AuditAction, AuditObservation

# 🎨 AegisOpenEnv: The Financial Compliance RL Sandbox
# Optimized for the Meta OpenEnv Prize Pool
app = create_fastapi_app(AegisOpenEnv, AuditAction, AuditObservation)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
