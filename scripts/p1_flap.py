"""P1 (GPU): FLAP / Wanda-SP / Magnitude-SP on the base model -> plans, perplexity, latency.

Steps (resumable: finished rows in sweep.csv and existing plan files are skipped):
  1. load base (fp16), WikiText-2 calibration (1024 x 128 tokens, seed 0) + test set
  2. FLAP scores in ONE calibration pass on the unpruned model -> scores.safetensors
  3. for every (method, ratio): plan -> plans/<method>_<ratio>.safetensors
     apply plan to a fresh copy of the base -> WikiText-2 PPL (seqlen 128, as in FLAP),
     parameter counts, batch-1 latency/energy profile
  4. copy plans to ${QACR_DRIVE_ROOT}/plans/ (stable path used by P2 outcome runs)

Outputs in results/p1/<run_id>/: scores.safetensors, plans/, sweep.csv, profiles.json, summary.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
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
from qacr.pruning.flap import (PrunePlan, apply_plan, eval_ppl, flap_plan, flap_scores,  # noqa: E402
                               magnitude_sp_plan, unpruned_plan, wanda_sp_plan, wikitext2)
from qacr.runtime.variants import parse_dtype  # noqa: E402

FIELDS = ["method", "ratio", "achieved_ratio", "n_params", "n_params_non_embedding", "heads_kept", "channels_kept",
          "min_heads_in_a_layer", "ppl_wikitext2", "prefill_ms_128", "prefill_ms_512", "prefill_ms_1024",
          "decode_ms_per_tok", "gen_tokens_per_s", "joules_per_token", "peak_mem_gb", "status"]


def tag(method: str, ratio: float) -> str:
    return f"{method}_{ratio:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p1", run.path("run.log"))
    device = get_dotted(cfg, "device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if get_dotted(cfg, "require_gpu", False) and device != "cuda":
        log.error("require_gpu=true but no CUDA device")
        close_logger(log)
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_id = cfg["base"]["path"]
    dtype = parse_dtype(cfg["base"].get("dtype", "float16"))
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).to(device).eval()
    log.info("base %s loaded on %s (%s)", base_id, device, dtype)

    cal = cfg["calibration"]
    t0 = time.perf_counter()
    calib, testenc = wikitext2(tok, cal["nsamples"], cal["seed"], cal["seqlen"])
    log.info("wikitext2: %d calibration windows x %d tokens; test %d tokens (%.0fs)", len(calib), cal["seqlen"],
             testenc.numel(), time.perf_counter() - t0)
    calib_wanda = calib[: get_dotted(cfg, "baselines.wanda_nsamples", 128)]

    plans_dir = run.path("plans")
    plans_dir.mkdir(exist_ok=True)
    scores_path = run.path("scores.safetensors")
    scores = None
    flap_ratios = [float(r) for r in cfg["flap_ratios"]]
    if flap_ratios:
        from safetensors.torch import load_file, save_file
        if scores_path.exists():
            raw = load_file(str(scores_path))
            meta = json.loads(run.path("scores_meta.json").read_text())
            scores = {**raw, **meta}
            log.info("loaded existing FLAP scores")
        else:
            t0 = time.perf_counter()
            scores = flap_scores(base, calib, device, log=log.info)
            save_file({k: v.contiguous() for k, v in scores.items() if torch.is_tensor(v)}, str(scores_path))
            run.write_json("scores_meta.json", {"head_dim": scores["head_dim"], "n_heads": scores["n_heads"],
                                                "seconds": time.perf_counter() - t0})
            log.info("FLAP scores computed in %.1fs", time.perf_counter() - t0)

    jobs = [("none", 0.0)] + [("flap", r) for r in flap_ratios]
    jobs += [(m, float(r)) for m in get_dotted(cfg, "baselines.methods", []) for r in cfg["baselines"]["ratios"]]

    csv_path = run.path("sweep.csv")
    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            done = {(r["method"], float(r["ratio"])) for r in csv.DictReader(f) if r["status"] == "ok"}
    profiles = json.loads(run.path("profiles.json").read_text()) if run.path("profiles.json").exists() else {}
    meter = EnergyMeter() if device == "cuda" else None
    prof_cfg = cfg.get("profile", {})

    for method, ratio in jobs:
        if (method, ratio) in done:
            log.info("[%s] done, skipping", tag(method, ratio))
            continue
        name = tag(method, ratio)
        plan_path = plans_dir / f"{name}.safetensors"
        t0 = time.perf_counter()
        if plan_path.exists():
            plan = PrunePlan.load(plan_path)
        elif method == "none":
            plan = unpruned_plan(base)
        elif method == "flap":
            plan = flap_plan(base, scores, ratio, meta={"base": base_id, "calibration": cal})
        elif method == "wanda_sp":
            plan = wanda_sp_plan(base, calib_wanda, ratio, device, log=log.info)
            plan.meta.update({"base": base_id, "calibration": {**cal, "nsamples": len(calib_wanda)}})
        elif method == "mag_sp":
            plan = magnitude_sp_plan(base, ratio)
            plan.meta.update({"base": base_id})
        else:
            raise ValueError(method)
        plan.save(plan_path)
        row = {"method": method, "ratio": ratio, "status": "ok",
               "achieved_ratio": plan.meta.get("achieved_ratio", 0.0),
               "heads_kept": int(plan.head_keep.sum()), "channels_kept": int(plan.chan_keep.sum()),
               "min_heads_in_a_layer": int(plan.head_keep.sum(dim=1).min())}
        try:
            model = apply_plan(copy.deepcopy(base), plan, physical=True).eval()
        except NotImplementedError as e:
            log.warning("[%s] cannot materialise: %s", name, e)
            row["status"] = f"skipped: {e}"
            _append(csv_path, row)
            continue
        row["n_params"] = sum(p.numel() for p in model.parameters())
        row["n_params_non_embedding"] = row["n_params"] - model.get_input_embeddings().weight.numel()
        row["ppl_wikitext2"] = eval_ppl(model, testenc, seqlen=cal["seqlen"], bs=get_dotted(cfg, "ppl.batch_size", 16),
                                        device=device)
        prof = profile_model(model, device, model.config.vocab_size,
                             prompt_lens=tuple(prof_cfg.get("prompt_lens", [128, 512, 1024])),
                             gen_tokens=prof_cfg.get("gen_tokens", 64), warmup=prof_cfg.get("warmup", 3),
                             reps=prof_cfg.get("reps", 10), seed=cfg["seed"], energy_meter=meter)
        profiles[name] = prof
        run.write_json("profiles.json", profiles)
        for L in (128, 512, 1024):
            row[f"prefill_ms_{L}"] = prof["prefill"].get(L, {}).get("median_ms")
        row["decode_ms_per_tok"] = prof["generate"]["decode_ms_per_tok"]
        row["gen_tokens_per_s"] = prof["generate"]["tokens_per_s"]
        row["joules_per_token"] = (prof["energy"] or {}).get("joules_per_generated_token")
        row["peak_mem_gb"] = prof.get("peak_mem_gb")
        _append(csv_path, row)
        log.info("[%s] achieved=%.3f params=%.3fB ppl=%.2f decode=%s ms/tok (%.0fs)", name, row["achieved_ratio"],
                 row["n_params"] / 1e9, row["ppl_wikitext2"], _fmt(row["decode_ms_per_tok"]), time.perf_counter() - t0)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if meter is not None:
        meter.close()

    drive = get_dotted(cfg, "paths.plans_export")
    if drive:
        Path(drive).mkdir(parents=True, exist_ok=True)
        for p in plans_dir.glob("*.safetensors"):
            shutil.copy2(p, Path(drive) / p.name)
        log.info("plans copied to %s", drive)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    run.write_json("summary.json", {"rows": rows})
    log.info("done: %d rows", len(rows))
    close_logger(log)
    return 0


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


if __name__ == "__main__":
    raise SystemExit(main())
