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


def power_state() -> dict:
    """Windows only: AC/battery status and current CPU performance, so CPU timings carry their conditions.
    Win32_Battery.BatteryStatus: 1 = discharging (on battery), 2 = on AC power (see Microsoft docs)."""
    import subprocess
    cmd = ("$b=(Get-CimInstance Win32_Battery | Select-Object -First 1 -ExpandProperty BatteryStatus);"
           "$p=Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\";"
           "Write-Output \"$b,$($p.PercentProcessorPerformance),$($p.ProcessorFrequency)\"")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=30).stdout.strip().splitlines()[-1]
        b, perf, freq = out.split(",")
        return {"battery_status": int(b) if b else None, "on_ac_power": (b == "2") if b else None,
                "cpu_percent_performance": float(perf), "cpu_frequency_mhz": float(freq)}
    except Exception as e:  # non-Windows or no battery
        return {"error": repr(e)}


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
        quant = get_dotted(cfg, "quantize", "none")
        if quant == "int8":
            from qacr.runtime.fast import quantize_int8_cpu
            model = quantize_int8_cpu(model)
        elif quant != "none":
            raise ValueError(f"unknown quantize option {quant}")
        ppl = None
        n_win = get_dotted(cfg, "ppl_windows")
        if n_win:
            from transformers import AutoTokenizer

            from qacr.pruning.flap import eval_ppl, wikitext2
            tok = AutoTokenizer.from_pretrained(cfg["base"]["path"])
            _, testenc = wikitext2(tok, 1, 0, 128)
            ppl = eval_ppl(model, testenc, seqlen=128, bs=8, device=device, max_windows=int(n_win))
        before = power_state() if device == "cpu" else None
        if get_dotted(cfg, "require_ac_power", False) and before and before.get("on_ac_power") is False:
            log.error("on battery power: CPU timings would not be comparable; plug in the charger")
            close_logger(log)
            return 3
        r = profile_model(model, device, model.config.vocab_size, prompt_lens=tuple(prof["prompt_lens"]),
                          gen_tokens=prof["gen_tokens"], warmup=prof["warmup"], reps=prof["reps"], seed=cfg["seed"],
                          energy_meter=meter)
        r["power_before"], r["power_after"] = before, (power_state() if device == "cpu" else None)
        r["quantize"], r["torch_threads"] = quant, torch.get_num_threads()
        r["ppl_wikitext2_first_windows"] = {"windows": n_win, "ppl": ppl} if n_win else None
        results[name] = r
        run.write_json("profiles.json", results)
        from qacr.runtime.fast import linear_weight_bytes
        row = {"variant": name, "quantize": quant, "threads": torch.get_num_threads(), "ppl_subset": ppl,
               "linear_weight_gb": linear_weight_bytes(model) / 1024 ** 3,
               "n_params": r["n_params"], "weight_gb": r["weight_bytes"] / 1024 ** 3,
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
