"""P2 step 3 (CPU): outcome matrix + headroom report (the P2 decision gate), per cost model.

Inputs (config):
  outcomes: {variant_name: <p2 outcome run dir containing outcomes.parquet>}
  profiles: {device_label: {"source": "p1_flap" | "p1_profile", "dir": <run dir>}}
  variant_profile_names: {variant_name: <name used in the profile files, e.g. flap_0.20>}
  splits: which pool splits to analyse (test by default)
Outputs in results/p2/<run_id>/: matrix.parquet, headroom_<device>.json, accuracy_table.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.eval.headroom import analyze, query_costs  # noqa: E402


def load_profiles(spec: dict, names: dict[str, str]) -> dict[str, dict]:
    """spec: {"dir": run_dir} (one profile run for all variants) or additionally
    {"per_variant_dirs": {variant: run_dir}} for variants profiled in other runs."""
    cache: dict[str, dict] = {}
    out = {}
    for variant, pname in names.items():
        d = (spec.get("per_variant_dirs") or {}).get(variant, spec["dir"])
        if d not in cache:
            cache[d] = json.loads((Path(d) / "profiles.json").read_text(encoding="utf-8"))
        p = cache[d][pname]
        lens = sorted(int(k) for k in p["prefill"])
        out[variant] = {"prefill_lengths": lens, "prefill_ms": [p["prefill"][str(L)]["median_ms"]
                                                                if str(L) in p["prefill"] else p["prefill"][L]["median_ms"]
                                                                for L in lens],
                        "decode_ms_per_tok": p["generate"]["decode_ms_per_tok"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p2.analyze", run.path("run.log"))

    pool = pd.read_parquet(REPO / get_dotted(cfg, "pool.path", "data/p2/query_pool.parquet"))
    splits = list(cfg.get("splits", ["test"]))
    keep = set(pool.loc[pool["split"].isin(splits), "query_id"])
    order = list(cfg["outcomes"].keys())
    frames = []
    for v, d in cfg["outcomes"].items():
        df = pd.read_parquet(Path(d) / "outcomes.parquet")
        got = set(df["variant"].unique())
        if got != {v}:
            raise ValueError(f"{d}: contains variants {got}, expected {v}")
        frames.append(df[df["query_id"].isin(keep)])
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(pool[["query_id", "n_choices"]], on="query_id", how="left", validate="many_to_one")
    log.info("outcomes: %d rows, %d queries, variants %s, splits %s", len(out), out["query_id"].nunique(), order, splits)

    # GSM8K prompts (few-shot exemplars) must be identical across variants
    ph = out.groupby("query_id")["prompt_hash"].nunique()
    if (ph > 1).any():
        raise ValueError(f"{int((ph > 1).sum())} queries have different prompts across variants")

    acc = out.pivot_table(index="task", columns="variant", values="correct", aggfunc="mean")[order]
    acc.loc["macro_avg"] = acc.mean()
    acc.to_csv(run.path("accuracy_table.csv"))
    log.info("accuracy (split=%s):\n%s", splits, acc.round(4).to_string())

    # Behaviour evidence: answer-distribution collapse (MC) and generation length (GSM8K)
    behaviour = {"mc_max_answer_share": {}, "gsm8k_generation": {}}
    for (t, v), g in out[out["task"] != "gsm8k"].groupby(["task", "variant"]):
        share = g["pred"].astype("Int64").value_counts(normalize=True)
        behaviour["mc_max_answer_share"].setdefault(t, {})[v] = {
            "max_share": float(share.max()), "answer": int(share.idxmax()), "accuracy": float(g["correct"].mean())}
    cap = int(cfg.get("gsm8k_max_gen_toks", 256))
    for v, g in out[out["task"] == "gsm8k"].groupby("variant"):
        behaviour["gsm8k_generation"][v] = {
            "accuracy": float(g["correct"].mean()), "mean_tokens": float(g["n_gen_tokens"].mean()),
            "median_tokens": float(g["n_gen_tokens"].median()),
            "share_at_cap": float((g["n_gen_tokens"] >= cap - 1).mean()), "cap": cap}
    run.write_json("behaviour.json", behaviour)

    names = cfg["variant_profile_names"]
    for label, spec in cfg["profiles"].items():
        prof = load_profiles(spec, names)
        df = out.copy()
        df["cost_ms"] = query_costs(df, prof)
        rep = analyze(df, order=order)
        rep.update({"cost_model": label, "splits": splits, "profile_dir": spec["dir"],
                    "note": "MC tasks: cost = prefill(n_ctx); GSM8K: prefill + n_gen * decode"})
        run.write_json(f"headroom_{label}.json", rep)
        log.info("[%s] static=%s", label, {v: (round(s['accuracy'], 4), round(s['cost_ms'], 1))
                                           for v, s in rep["static"].items()})
        log.info("[%s] oracle acc=%.4f cost=%.1fms | hull@oracle-cost=%.4f | headroom=%s | unsolvable=%.3f",
                 label, rep["oracle"]["accuracy"], rep["oracle"]["cost_ms"], rep["hull_accuracy_at_oracle_cost"],
                 rep["headroom_points"], rep["oracle"]["unsolvable_share"])
        df.to_parquet(run.path(f"matrix_{label}.parquet"), index=False)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
