"""Run directory management: every script writes to results/<phase>/<run_id>/.

A run directory always contains:
  config.yaml   - the fully resolved config (after overrides)
  meta.json     - run id, git commit, start time, seed, command line
  env.json      - python / library / hardware versions
  pip_freeze.txt
  run.log       - log file (see logging_utils)
Passing an existing ``run_id`` resumes into the same directory instead of creating a new one.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import dump_config, get_dotted

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def git_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def env_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    for mod in ("numpy", "torch", "transformers", "datasets", "accelerate", "lm_eval",
                "sentence_transformers", "sklearn", "huggingface_hub", "pynvml"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # not installed in this environment
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / 1024**3, 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info


def _pip_freeze() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"pip freeze failed: {e}\n"


@dataclass
class Run:
    run_id: str
    dir: Path
    cfg: dict
    resumed: bool

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        tmp.replace(p)  # atomic on the same filesystem: no half-written result files
        return p


def make_run_id(run_name: str, now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-") or "run"
    return f"{stamp}_{safe}"


def results_root(cfg: dict) -> Path:
    root = get_dotted(cfg, "paths.results_root") or str(REPO_ROOT / "results")
    return Path(root)


def start_run(cfg: dict, run_id: str | None = None, argv: list[str] | None = None) -> Run:
    """Create (or resume) the run directory and write config/meta/env files."""
    phase = str(cfg["phase"])
    if run_id is not None and not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id {run_id!r}")
    resumed = run_id is not None
    run_id = run_id or make_run_id(cfg["run_name"])
    run_dir = results_root(cfg) / phase / run_id
    if resumed and not run_dir.exists():
        raise FileNotFoundError(f"Asked to resume {run_id} but {run_dir} does not exist")
    if not resumed and run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; pass its run_id to resume")
    run_dir.mkdir(parents=True, exist_ok=True)

    run = Run(run_id=run_id, dir=run_dir, cfg=cfg, resumed=resumed)
    if resumed:
        # Never silently change the experiment on resume.
        old_cfg_path = run_dir / "config.yaml"
        if old_cfg_path.exists():
            import yaml
            with old_cfg_path.open(encoding="utf-8") as f:
                old_cfg = yaml.safe_load(f)
            if old_cfg != cfg:
                raise ValueError(f"Config differs from the one stored in {old_cfg_path}; refusing to resume")
    else:
        dump_config(cfg, run_dir / "config.yaml")
    meta_name = "meta.json" if not resumed else f"meta_resume_{make_run_id('r')}.json"
    run.write_json(meta_name, {
        "run_id": run_id,
        "phase": phase,
        "seed": cfg["seed"],
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "argv": argv if argv is not None else sys.argv,
        "git": git_commit(),
        "resumed": resumed,
    })
    run.write_json("env.json", env_snapshot())
    (run_dir / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    return run
