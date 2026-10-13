"""YAML config loading with dotted-key CLI overrides.

Every script takes ``--config path.yaml`` plus optional ``--set a.b=value`` overrides.
Values in overrides are parsed as YAML scalars, so ``--set seed=3`` gives an int and
``--set paths.results_root=/x`` gives a string.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get_dotted(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got: {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty key in override: {text!r}")
    return key, yaml.safe_load(raw) if raw != "" else ""


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_str(s: str) -> str | None:
    """Expand ${VAR} and ${VAR:-default}. Unset variables expand to '' (or the default).
    A string that expands to '' becomes None, so "unset" means "use the code default"."""
    if "${" not in s:
        return s
    out = _ENV_RE.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), s)
    return out if out != "" else None


def _expand_env(obj: Any) -> Any:
    """Expand environment variables in string values (used for Drive paths set by the notebook)."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _expand_str(obj)
    return obj


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        key, value = parse_override(ov)
        _set_dotted(cfg, key, value)
    cfg = _expand_env(cfg)
    for required in ("phase", "run_name", "seed"):
        if required not in cfg:
            raise KeyError(f"Config {path} is missing required key '{required}'")
    return cfg


def dump_config(cfg: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
