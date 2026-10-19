"""P1 (edge device): batch-1 latency profile of the base model and its pruning plans on THIS machine.

Used for the laptop-CPU measurements. For each variant the base model is re-loaded from disk and the
plan applied in place (never two copies in RAM). Energy is recorded only where NVML exists (GPU);
on a CPU it is reported as unavailable.

Outputs in results/p1/<run_id>/: profiles.json, profile.csv
Usage: python scripts/p1_profile.py --config configs/p1_profile_cpu.yaml --set plans_dir=<dir with plans>
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.energy import EnergyMeter  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.profiling.latency import profile_model  # noqa: E402
from qacr.pruning.flap import PrunePlan, apply_plan  # noqa: E402
from qacr.runtime.variants import parse_dtype  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p1.profile", run.path("run.log"))

    device = get_dotted(cfg, "device", "cpu")
    threads = get_dotted(cfg, "threads")
    if threads:
        torch.set_num_threads(int(threads))
    plans_dir = Path(cfg["plans_dir"])
    dtype = parse_dtype(cfg["base"].get("dtype", "float32"))
    prof = cfg["profile"]
    hw = {"device": device, "torch_threads": torch.get_num_threads(), "processor": platform.processor(),
          "platform": platform.platform(), "dtype": str(dtype)}
    run.write_json("hardware.json", hw)
    log.info("hardware: %s", hw)

    from transformers import AutoModelForCausalLM
    meter = EnergyMeter() if device.startswith("cuda") else None
    results, rows = {}, []
    for name in cfg["variants"]:
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(cfg["base"]["path"], dtype=dtype).to(device).eval()
        if name != "none_0.00":
            apply_plan(model, PrunePlan.load(plans_dir / f"{name}.safetensors"), physical=True)
        r = profile_model(model, device, model.config.vocab_size, prompt_lens=tuple(prof["prompt_lens"]),
                          gen_tokens=prof["gen_tokens"], warmup=prof["warmup"], reps=prof["reps"], seed=cfg["seed"],
                          energy_meter=meter)
        results[name] = r
        run.write_json("profiles.json", results)
        row = {"variant": name, "n_params": r["n_params"], "weight_gb": r["weight_bytes"] / 1024 ** 3,
               "decode_ms_per_tok": r["generate"]["decode_ms_per_tok"], "gen_tokens_per_s": r["generate"]["tokens_per_s"]}
        for L in prof["prompt_lens"]:
            row[f"prefill_ms_{L}"] = r["prefill"][L]["median_ms"]
        rows.append(row)
        log.info("[%s] params=%.3fB decode=%.1f ms/tok prefill=%s (%.0fs)", name, r["n_params"] / 1e9,
                 row["decode_ms_per_tok"], {L: round(r["prefill"][L]["median_ms"], 1) for L in prof["prompt_lens"]},
                 time.perf_counter() - t0)
        del model
    with run.path("profile.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("done")
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
