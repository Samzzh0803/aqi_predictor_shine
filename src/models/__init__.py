"""Model helpers and local fallback registry interfaces."""

from src.models.registry import (
    MODEL_NAME,
    RegisteredModelVersion,
    get_champion,
    list_registered_versions,
    register_model_version,
)

__all__ = [
    "MODEL_NAME",
    "RegisteredModelVersion",
    "get_champion",
    "list_registered_versions",
    "register_model_version",
]
