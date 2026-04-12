"""
FastAPI application entry point for the Compliance environment.

This module provides the standard server entry point used by OpenEnv tooling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    REPO_ROOT = SERVER_DIR.parent
except IndexError:
    REPO_ROOT = None

if REPO_ROOT is not None:
    SRC_DIR = REPO_ROOT
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from .server import ComplianceEnv
from openenv.core.env_server import create_fastapi_app
from .models import ComplianceAction, ComplianceObservation

app = create_fastapi_app(ComplianceEnv, ComplianceAction, ComplianceObservation)


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the Compliance environment server with uvicorn."""

    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
