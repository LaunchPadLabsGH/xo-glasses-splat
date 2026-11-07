"""Configuration loading: default YAML + device profile + `key.sub=value` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` merged recursively into `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse `a.b.c=value`; the value is read as YAML so numbers/bools/null work."""
    if "=" not in text:
        raise ValueError(f"override must look like key.sub=value, got {text!r}")
    key, raw = text.split("=", 1)
    path = [p for p in key.strip().split(".") if p]
    if not path:
        raise ValueError(f"empty key in override {text!r}")
    return path, yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for text in overrides:
        path, value = parse_override(text)
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return out


def load_device(name_or_path: str, config_dir: Path = CONFIG_DIR) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = config_dir / "devices" / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"device profile not found: {path}")
    with open(path) as f:
        dev = yaml.safe_load(f) or {}
    dev.setdefault("id", path.stem)
    return dev


def load_config(
    config_path: str | Path | None = None,
    device: str | None = None,
    overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict:
    """Build the effective config. Precedence: overrides > config file > default.yaml."""
    with open(config_dir / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if config_path:
        with open(config_path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides or [])
    if device:
        cfg["device"] = device
    cfg["device_profile"] = load_device(cfg["device"], config_dir)
    return cfg
