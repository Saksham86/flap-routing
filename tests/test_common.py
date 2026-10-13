import json

import numpy as np
import pytest
import torch
import yaml

from qacr.common.config import get_dotted, load_config, parse_override
from qacr.common.energy import EnergyMeter
from qacr.common.run import make_run_id, start_run
from qacr.common.seed import seed_everything


def _write_cfg(tmp_path, extra=None):
    cfg = {"phase": "p0", "run_name": "unit test", "seed": 7,
           "paths": {"results_root": str(tmp_path / "results")}, "a": {"b": 1}}
    cfg.update(extra or {})
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ---- config ----------------------------------------------------------------------------
def test_override_types():
    assert parse_override("seed=3") == ("seed", 3)
    assert parse_override("x.y=0.5") == ("x.y", 0.5)
    assert parse_override("flag=true") == ("flag", True)
    assert parse_override("p=/content/drive") == ("p", "/content/drive")
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_load_config_overrides_and_required(tmp_path):
    cfg = load_config(_write_cfg(tmp_path), ["a.b=5", "new.key=hi"])
    assert get_dotted(cfg, "a.b") == 5 and get_dotted(cfg, "new.key") == "hi"
    bad = tmp_path / "bad.yaml"
    bad.write_text("phase: p0\nseed: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)


def test_env_expansion(tmp_path, monkeypatch):
    p = _write_cfg(tmp_path, {"paths": {"results_root": "${QACR_TEST_ROOT}", "other": "${QACR_UNSET_X:-fallback}",
                                        "mixed": "${QACR_TEST_ROOT}/sub"}})
    monkeypatch.delenv("QACR_TEST_ROOT", raising=False)
    monkeypatch.delenv("QACR_UNSET_X", raising=False)
    cfg = load_config(p)
    assert cfg["paths"]["results_root"] is None          # unset -> None -> code default
    assert cfg["paths"]["other"] == "fallback"
    monkeypatch.setenv("QACR_TEST_ROOT", "/drive/x")
    cfg = load_config(p)
    assert cfg["paths"]["results_root"] == "/drive/x" and cfg["paths"]["mixed"] == "/drive/x/sub"


# ---- seeding ---------------------------------------------------------------------------
def test_seed_reproducible():
    seed_everything(11)
    a = (np.random.rand(5), torch.randn(5))
    seed_everything(11)
    b = (np.random.rand(5), torch.randn(5))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])
    with pytest.raises(TypeError):
        seed_everything(1.5)


# ---- run directories -------------------------------------------------------------------
def test_run_id_sanitised():
    rid = make_run_id("my run/../x")
    assert "/" not in rid and " " not in rid and rid.endswith("my-run-..-x")


def test_start_run_and_resume(tmp_path):
    cfg = load_config(_write_cfg(tmp_path))
    run = start_run(cfg, argv=["test"])
    for f in ("config.yaml", "meta.json", "env.json", "pip_freeze.txt"):
        assert (run.dir / f).exists(), f
    assert run.dir.parent.name == "p0"
    meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7 and meta["resumed"] is False
    run.write_json("x.json", {"v": 1})
    assert json.loads((run.dir / "x.json").read_text(encoding="utf-8")) == {"v": 1}

    resumed = start_run(cfg, run_id=run.run_id, argv=["test"])
    assert resumed.dir == run.dir and resumed.resumed
    with pytest.raises(ValueError):            # changed config must not resume silently
        start_run(dict(cfg, seed=8), run_id=run.run_id)
    with pytest.raises(FileNotFoundError):
        start_run(cfg, run_id="does-not-exist")
    with pytest.raises(ValueError):
        start_run(cfg, run_id="../escape")


# ---- energy ----------------------------------------------------------------------------
def test_energy_meter_never_raises():
    m = EnergyMeter()
    m.start()
    r = m.stop()
    assert r.duration_s >= 0
    if not m.available:
        assert r.available is False and r.notes
    m.close()
