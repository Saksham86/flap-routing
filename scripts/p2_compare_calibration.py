"""P2 (CPU): WikiText-calibrated FLAP vs in-domain-calibrated FLAP at matched ratios, on the test split.

For each ratio: macro accuracy of both variants, the paired difference with a 95% bootstrap CI (questions resampled
within each task, macro = mean over tasks), McNemar's exact test on all questions, per-task accuracies, and two
collapse indicators (share of the most frequent MMLU / BoolQ answer; share of GSM8K answers hitting the token cap).
WikiText-2 perplexities are read from the two P1 sweep files.

Outputs in results/p2/<run_id>/: calibration_comparison.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from qacr.common.config import load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402


def sweep_ppl(run_dir: str) -> dict:
    with (REPO / run_dir / "sweep.csv").open(encoding="utf-8") as f:
        return {(r["method"], round(float(r["ratio"]), 2)): float(r["ppl_wikitext2"]) for r in csv.DictReader(f)
                if r["status"] == "ok"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p2.calib", run.path("run.log"))
    cap = int(cfg.get("gsm8k_cap", 256))

    df = pd.concat([pd.read_parquet(REPO / d / "outcomes.parquet") for d in cfg["outcomes"].values()], ignore_index=True)
    if (df.groupby("query_id")["prompt_hash"].nunique() > 1).any():
        raise ValueError("prompts differ across variants")
    ppl = {**sweep_ppl(cfg["sweeps"]["wikitext"]), **sweep_ppl(cfg["sweeps"]["indomain"])}
    rng = np.random.default_rng(int(cfg["seed"]))
    out = {"ratios": {}, "n_questions": int(df["query_id"].nunique())}

    def collapse(v):
        x = df[df["variant"] == v]
        mc = {t: float(x[x["task"] == t]["pred"].astype("Int64").value_counts(normalize=True).max())
              for t in ("mmlu", "boolq")}
        g = x[x["task"] == "gsm8k"]
        return {**{f"{t}_max_answer_share": s for t, s in mc.items()},
                "gsm8k_share_at_cap": float((g["n_gen_tokens"] >= cap - 1).mean())}

    out["reference"] = {"variant": cfg["reference"],
                        "per_task": df[df["variant"] == cfg["reference"]].groupby("task")["correct"].mean().to_dict(),
                        **collapse(cfg["reference"])}
    out["reference"]["macro"] = float(np.mean(list(out["reference"]["per_task"].values())))
    for r in cfg["ratios"]:
        a_name, b_name = f"flap_{r:.2f}", f"flaptask_{r:.2f}"
        A = df[df["variant"] == a_name].set_index("query_id").sort_index()
        B = df[df["variant"] == b_name].set_index("query_id").sort_index()
        if not A.index.equals(B.index):
            raise ValueError(f"{a_name} and {b_name} cover different questions")
        task = A["task"].to_numpy()
        a, b = A["correct"].to_numpy(float), B["correct"].to_numpy(float)
        tasks = sorted(set(task))
        idx = {t: np.where(task == t)[0] for t in tasks}
        macro = lambda y: float(np.mean([y[idx[t]].mean() for t in tasks]))  # noqa: E731
        boots = []
        for _ in range(int(cfg["n_boot"])):
            samp = {t: idx[t][rng.integers(0, len(idx[t]), len(idx[t]))] for t in tasks}
            boots.append(np.mean([b[samp[t]].mean() - a[samp[t]].mean() for t in tasks]))
        fixed, broken = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
        res = {
            "wikitext": {"variant": a_name, "macro": macro(a), "per_task": {t: float(a[idx[t]].mean()) for t in tasks},
                         "ppl_wikitext2": ppl.get(("flap", round(r, 2))), **collapse(a_name)},
            "indomain": {"variant": b_name, "macro": macro(b), "per_task": {t: float(b[idx[t]].mean()) for t in tasks},
                         "ppl_wikitext2": ppl.get(("flaptask", round(r, 2))), **collapse(b_name)},
            "macro_diff": macro(b) - macro(a),
            "macro_diff_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "questions_fixed": fixed, "questions_broken": broken,
            "mcnemar_p": float(binomtest(fixed, fixed + broken, 0.5).pvalue),
        }
        out["ratios"][f"{r:.2f}"] = res
        log.info("ratio %.2f: macro %.2f -> %.2f (%+.2f, CI %+.2f..%+.2f), fixed %d broken %d, McNemar p=%.1e",
                 r, res["wikitext"]["macro"] * 100, res["indomain"]["macro"] * 100, res["macro_diff"] * 100,
                 res["macro_diff_ci95"][0] * 100, res["macro_diff_ci95"][1] * 100, fixed, broken, res["mcnemar_p"])
    run.write_json("calibration_comparison.json", out)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
