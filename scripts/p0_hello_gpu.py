"""P0 exit check: verify the environment end to end and write a result file.

Checks (each recorded in results/p0/<run_id>/hello.json, failures are recorded, not hidden):
  1. library versions + hardware                     (env.json)
  2. seeding reproduces identical random tensors
  3. matmul benchmark (fp16 on GPU / fp32 on CPU) with CUDA-synchronised timing
  4. NVML energy: idle power, counter vs sampler energy for the benchmark window
  5. pruned-shape smoke test (per-layer different head/channel counts + bias) on this device
  6. Hugging Face Hub reachable: download SmolLM2-1.7B config.json and check its architecture

Usage:
  python scripts/p0_hello_gpu.py --config configs/p0_hello.yaml [--set key=value ...] [--run-id ID]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.energy import EnergyMeter  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.common.smoke import pruned_shape_smoke_test  # noqa: E402


def check_seeding(seed: int) -> dict:
    hashes = []
    for _ in range(2):
        seed_everything(seed)
        t = torch.randn(1024)
        hashes.append(hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16])
    return {"hashes": hashes, "reproducible": hashes[0] == hashes[1]}


def matmul_benchmark(device: str, n: int, seconds: float, warmup: int, meter: EnergyMeter | None) -> dict:
    dtype = torch.float16 if device == "cuda" else torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        a @ b
    sync()
    if meter is not None:
        meter.start()
    iters = 0
    t0 = time.perf_counter()
    while True:
        a @ b
        iters += 1
        if iters % 10 == 0:
            sync()
            if time.perf_counter() - t0 >= seconds:
                break
    sync()
    elapsed = time.perf_counter() - t0
    energy = meter.stop().to_dict() if meter is not None else None
    tflops = 2 * n**3 * iters / elapsed / 1e12
    return {"device": device, "dtype": str(dtype), "n": n, "iters": iters, "elapsed_s": elapsed,
            "tflops": tflops, "energy": energy}


def check_hub(model_id: str) -> dict:
    import json

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_id, "config.json")
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    keys = ["architectures", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "attention_bias", "tie_word_embeddings", "vocab_size"]
    return {"model_id": model_id, "config": {k: c.get(k) for k in keys}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p0", run.path("run.log"))
    log.info("run dir: %s", run.dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if get_dotted(cfg, "require_gpu", False) and device != "cuda":
        log.error("require_gpu=true but no CUDA device is visible (Colab: Runtime > Change runtime type > T4 GPU)")
        run.write_json("hello.json", {"passed": False, "error": "no GPU"})
        close_logger(log)
        return 2

    results: dict = {"device": device, "checks": {}}

    def step(name, fn):
        log.info("check: %s", name)
        try:
            out = fn()
            results["checks"][name] = {"ok": True, **out} if isinstance(out, dict) else {"ok": True, "value": out}
        except Exception as e:  # record and continue: we want the full picture in one run
            results["checks"][name] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
            log.error("check %s failed: %r", name, e)

    step("seeding", lambda: check_seeding(cfg["seed"]))

    meter = EnergyMeter(gpu_index=0, sample_interval_s=get_dotted(cfg, "energy.sample_interval_s", 0.02))
    results["checks"]["nvml"] = {"ok": meter.available, "counter_supported": meter.counter_supported,
                                 "device_name": meter.device_name(), "reason": meter.reason}
    if meter.available:
        step("idle_power", lambda: meter.measure_idle(get_dotted(cfg, "energy.idle_seconds", 3.0)).to_dict())

    bench = cfg.get("bench", {})
    n = bench.get("n_gpu", 4096) if device == "cuda" else bench.get("n_cpu", 1024)
    step("matmul", lambda: matmul_benchmark(device, n, bench.get("seconds", 10.0), bench.get("warmup", 10),
                                            meter if meter.available else None))
    meter.close()

    dtypes = [torch.float32] + ([torch.float16] if device == "cuda" else [])
    for dt in dtypes:
        for impl in ("eager", "sdpa"):
            step(f"pruned_shapes_{impl}_{str(dt).split('.')[-1]}",
                 lambda dt=dt, impl=impl: pruned_shape_smoke_test(device=device, dtype=dt, attn_impl=impl, seed=cfg["seed"]))

    if get_dotted(cfg, "hub.check", True):
        step("hub", lambda: check_hub(get_dotted(cfg, "hub.model_id", "HuggingFaceTB/SmolLM2-1.7B")))

    # Overall verdict. NVML/energy is required only when running on GPU.
    required = ["seeding", "matmul"] + [k for k in results["checks"] if k.startswith("pruned_shapes")]
    if device == "cuda":
        required += ["nvml"]
    failed = [k for k in required if not results["checks"].get(k, {}).get("ok")]
    failed += [k for k in results["checks"] if k.startswith("pruned_shapes")
               and results["checks"][k].get("ok") and not results["checks"][k].get("passed")]
    if not results["checks"]["seeding"].get("reproducible", False):
        failed.append("seeding(reproducible)")
    results["failed"] = sorted(set(failed))
    results["passed"] = not failed
    out = run.write_json("hello.json", results)
    log.info("wrote %s", out)
    log.info("PASSED" if results["passed"] else f"FAILED: {results['failed']}")

    m = results["checks"].get("matmul", {})
    if m.get("ok"):
        e = m.get("energy") or {}
        log.info("matmul: %.2f TFLOPS over %.1fs | energy counter=%s J, sampled=%s J, mean power=%s W",
                 m["tflops"], m["elapsed_s"], _fmt(e.get("counter_joules")), _fmt(e.get("sampled_joules")),
                 _fmt(e.get("mean_power_w")))
    close_logger(log)
    return 0 if results["passed"] else 1


def _fmt(x):
    return "n/a" if x is None else f"{x:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
