"""Configuration loader for the pneumonia detection pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels up from backend/app/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve relative paths."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve path entries
    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str(_resolve_path(value))

    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_path"] = str(path)
    return config


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Cached config accessor. Uses CONFIG_PATH env var if set."""
    env_path = os.environ.get("CONFIG_PATH")
    return load_config(env_path)


def get_age_group(age: int | float, config: dict[str, Any] | None = None) -> str:
    """Map numeric age to configured age group label."""
    cfg = config or get_config()
    age_int = int(age)
    for group in cfg["age_groups"]:
        if group["min"] <= age_int <= group["max"]:
            return group["name"]
    return "unknown"


def ensure_directories(config: dict[str, Any] | None = None) -> None:
    """Create required output directories."""
    cfg = config or get_config()
    dirs = [
        cfg["paths"]["data_processed"],
        cfg["paths"]["data_train"],
        cfg["paths"]["data_val"],
        cfg["paths"]["data_test"],
        cfg["paths"]["metrics"],
        cfg["paths"]["plots"],
        cfg["paths"]["predictions"],
        Path(cfg["paths"]["baseline_model"]).parent,
        Path(cfg["paths"]["segmented_model"]).parent,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
