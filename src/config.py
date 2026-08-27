"""Typed configuration: YAML file + environment-variable secrets."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"


class Settings(BaseSettings):
    """Environment-variable secrets — never hardcoded, never logged."""

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    hopsworks_api_key: str = ""
    hopsworks_project: str = ""
    aqi_api_base: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load config.yaml; inject Hopsworks project from env if not set."""
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    settings = get_settings()
    if settings.hopsworks_project:
        cfg.setdefault("hopsworks", {})["project"] = settings.hopsworks_project
    return cfg
