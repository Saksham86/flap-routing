"""P1 speed-up experiment (GPU, Colab T4): does removing per-step overhead make pruning pay off?

For each variant (base + plan): batch-1 greedy generation of N tokens after a 128-token prompt in
  (a) eager mode with the default dynamic cache (what P1 measured: ~30.5 ms/token for every ratio), and
  (b) static KV cache + torch.compile(mode="reduce-overhead") (CUDA graphs).
Correctness: (b) must produce the same greedy tokens as (a) (fp16, reported as a match fraction).
Optionally also "aligned" FLAP plans: each layer's kept MLP channels rounded DOWN to a multiple of `align`
(keeping the highest-scoring channels, bias compensation recomputed), to test GPU-friendly shapes.

Outputs in results/p1/<run_id>/: fast_gpu.csv, fast_gpu.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import statistics
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
from qacr.pruning.flap import PrunePlan, apply_plan, compensation_bias, param_count  # noqa: E402


def aligned_plan(base, plan: PrunePlan, scores: dict, align: int) -> PrunePlan:
    """Round each layer's kept MLP channels down to a multiple of `align` (min `align`), keeping top scores."""
    from qacr.pruning.flap import _standardize
    mlp = _standardize(scores["mlp_metric"])
    ck = torch.zeros_like(plan.chan_keep)
    for i in range(ck.shape[0]):
        k = int(plan.chan_keep[i].sum())
        k = max(align, (k // align) * align)
        ck[i, torch.topk(mlp[i], k).indices] = True
    ab, mb = compensation_bias(base, plan.head_keep, ck, scores)
    L, H = plan.head_keep.shape
    hd = scores["head_dim"]
    meta = dict(plan.meta, aligned_to=align, **param_count(L, H, hd, base.config.hidden_size,
                                                          base.config.intermediate_size, plan.head_keep, ck))
    return PrunePlan(plan.method + f"_a{align}", plan.ratio, plan.head_keep.clone(), ck, ab, mb, meta=meta)


@torch.no_grad()
def time_generate(model, ids, n_new, warmup, reps, meter=None):
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, pad_token_id=0)
    out = None
    for _ in range(warmup):
        out = model.generate(ids, **kw)
    torch.cuda.synchronize()
    if meter is not None and meter.available:
        meter.start()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model.generate(ids, **kw)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    e = meter.stop().to_dict() if (meter is not None and meter.available) else None
    return out, ts, e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p1.fastgpu", run.path("run.log"))
    if not torch.cuda.is_available():
        log.error("needs a GPU")
        close_logger(log)
        return 2

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from qacr.runtime.fast import enable_static_compile

    plans_dir = Path(cfg["plans_dir"])
    base = AutoModelForCausalLM.from_pretrained(cfg["base"]["path"], dtype=torch.float16).cuda().eval()
    scores = None
    if cfg.get("align"):
        sc_dir = Path(cfg["scores_dir"])
        import json
        scores = {**load_file(str(sc_dir / "scores.safetensors")),
                  **json.loads((sc_dir / "scores_meta.json").read_text())}
    g = torch.Generator().manual_seed(cfg["seed"])
    ids = torch.randint(0, base.config.vocab_size, (1, cfg["prompt_len"]), generator=g).cuda()
    n_new, warm, reps = cfg["gen_tokens"], cfg["warmup"], cfg["reps"]
    meter = EnergyMeter()

    jobs = []
    for name in cfg["variants"]:
        plan = None if name == "none_0.00" else PrunePlan.load(plans_dir / f"{name}.safetensors")
        jobs.append((name, plan))
        if cfg.get("align") and plan is not None and plan.method == "flap":
            jobs.append((f"{name}_a{cfg['align']}", aligned_plan(base, plan, scores, int(cfg["align"]))))

    rows, details = [], {}
    for name, plan in jobs:
        for mode in ("eager", "static_compiled"):
            model = copy.deepcopy(base)
            if plan is not None:
                apply_plan(model, plan, physical=True)
            n_params = sum(p.numel() for p in model.parameters())
            if mode == "static_compiled":
                enable_static_compile(model, max_cache_len=cfg["prompt_len"] + n_new + 8)
            t0 = time.perf_counter()
            try:
                out, ts, e = time_generate(model, ids, n_new, warm, reps, meter)
            except Exception as ex:  # record and continue (e.g. compile failure)
                log.error("[%s/%s] failed: %r", name, mode, ex)
                rows.append({"variant": name, "mode": mode, "status": f"failed: {ex!r}"[:300]})
                del model
                torch.cuda.empty_cache()
                continue
            med = statistics.median(ts)
            row = {"variant": name, "mode": mode, "status": "ok", "n_params": n_params,
                   "gen_ms_median": med, "gen_ms_min": min(ts), "gen_ms_max": max(ts),
                   "ms_per_token": med / n_new, "tokens_per_s": n_new / (med / 1000),
                   "joules_per_token": (e.get("counter_joules") / (reps * n_new)) if e and e.get("counter_joules") else None,
                   "mean_power_w": e.get("mean_power_w") if e else None,
                   "achieved_ratio": (plan.meta.get("achieved_ratio") if plan is not None else 0.0),
                   "setup_s": time.perf_counter() - t0}
            details[f"{name}/{mode}"] = out[0, -n_new:].tolist()
            rows.append(row)
            log.info("[%s/%s] %.2f ms/token (%.1f tok/s) J/tok=%s", name, mode, row["ms_per_token"], row["tokens_per_s"],
                     row["joules_per_token"])
            del model
            torch.cuda.empty_cache()
    # correctness: compiled tokens vs eager tokens
    for r in rows:
        if r.get("mode") == "static_compiled" and r["status"] == "ok":
            a, b = details.get(f"{r['variant']}/eager"), details.get(f"{r['variant']}/static_compiled")
            if a and b:
                first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
                r["tokens_identical_prefix"] = first_diff
                r["token_match_fraction"] = sum(x == y for x, y in zip(a, b)) / len(a)
    meter.close()
    fields = sorted({k for r in rows for k in r})
    with run.path("fast_gpu.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    run.write_json("fast_gpu.json", {"rows": rows, "tokens": details})
    log.info("done")
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
