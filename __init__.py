"""ComplianceEnv package exports."""

from typing import Any

from .models import (
    ComplianceAction,
    ComplianceObservation,
)

__all__ = [
    "ComplianceAction",
    "ComplianceObservation",
    "ComplianceEnv",
]


def __getattr__(name: str) -> Any:
    if name == "ComplianceEnv":
        from .client import ComplianceEnv as _ComplianceEnv

        return _ComplianceEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
