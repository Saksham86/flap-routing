"""P6 (CPU, offline): query-level confidence cascade on the stored outcome matrix.

For each cheap variant and each accuracy-loss budget: tau chosen on the calibration half, results on the
evaluation half (src/qacr/eval/cascade.py). Cost per question uses the measured T4 profiles (same cost model as
P2); the parameter ratio is also reported as a device-independent reference, clearly labelled as such.

Outputs in results/p6/<run_id>/: cascade.json, cascade.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from qacr.common.config import load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.eval.cascade import break_even, evaluate_cascade, option_confidence  # noqa: E402
from qacr.eval.headroom import query_costs  # noqa: E402


def load_profile(run_dir: str, name: str) -> dict:
    p = json.loads((REPO / run_dir / "profiles.json").read_text(encoding="utf-8"))[name]
    lens = sorted(int(k) for k in p["prefill"])
    return {"prefill_lengths": lens, "prefill_ms": [p["prefill"][str(L)]["median_ms"] for L in lens],
            "decode_ms_per_tok": p["generate"]["decode_ms_per_tok"], "n_params": p["n_params"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p6.cascade", run.path("run.log"))

    pool = pd.read_parquet(REPO / cfg.get("pool_path", "data/p2/query_pool.parquet")).set_index("query_id")
    frames = []
    for v, spec in cfg["variants"].items():
        df = pd.read_parquet(REPO / spec["outcomes"] / "outcomes.parquet")
        if set(df["variant"]) != {v}:
            raise ValueError(f"{spec['outcomes']}: variant mismatch")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["query_id"].map(pool["split"]) == "test"]
    mc = out[out["task"] != "gsm8k"].copy()
    conf = [option_confidence(r.lls, pool.at[r.query_id, "lm_choices"], pool.at[r.query_id, "metric"])
            for r in mc.itertuples()]
    mc["conf"] = [c[0] for c in conf]
    mc["margin"] = [c[1] for c in conf]
    profiles = {v: load_profile(s["profile_dir"], s["profile_name"]) for v, s in cfg["variants"].items()}
    mc["cost_ms"] = query_costs(mc, profiles)
    log.info("MC questions per variant: %s", mc.groupby("variant").size().to_dict())

    full = cfg["full"]
    results, rows = [], []
    for small in cfg["cheap"]:
        for max_loss in cfg["max_loss"]:
            r = evaluate_cascade(mc, small, full, max_loss, cost_col="cost_ms")
            r["param_ratio_small_over_full"] = profiles[small]["n_params"] / profiles[full]["n_params"]
            r["break_even_param_ratio"] = break_even(r["evaluation"]["served"], r["param_ratio_small_over_full"])
            results.append(r)
            e = r["evaluation"]
            row = {"small": small, "max_loss": max_loss, "tau": r["tau"], "served": e["served"],
                   "full_accuracy": e["full_accuracy"], "cascade_accuracy": e["accuracy"], "loss": e["loss"],
                   "loss_ci_lo": e["loss_ci95"][0], "loss_ci_hi": e["loss_ci95"][1],
                   "t4_cost_ratio": e["cost_ratio_small_over_full"], "t4_cascade_cost": e["cascade_cost_over_full"],
                   "param_ratio": r["param_ratio_small_over_full"],
                   "param_cascade_cost": r["break_even_param_ratio"]["relative_cost"]}
            rows.append(row)
            log.info("[%s, loss<=%.3f] tau=%.3f served=%.1f%% acc %.2f%% vs %.2f%% (loss %.2f, CI %.2f..%.2f) | "
                     "T4 cascade cost %.2fx | param-ratio cascade cost %.2fx", small, max_loss, r["tau"],
                     e["served"] * 100, e["accuracy"] * 100, e["full_accuracy"] * 100, e["loss"] * 100,
                     e["loss_ci95"][0] * 100, e["loss_ci95"][1] * 100, e["cascade_cost_over_full"],
                     r["break_even_param_ratio"]["relative_cost"])
    run.write_json("cascade.json", {"full": full, "results": results})
    with run.path("cascade.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
