"""Outcome-matrix analysis: static variants, random-mixture hull, per-query oracle, headroom.

Definitions (all on one query set Q, e.g. the test split):
  Y[q, v]      1 if variant v answers q correctly (primary metric of q's task)
  c(q, v)      cost of answering q with v. Latency cost model from measured profiles:
                 prefill_v(n_ctx(q)) + n_gen(q, v) * decode_v
               prefill is linearly interpolated between the measured prompt lengths
               (extrapolated linearly beyond them); n_gen = 0 for log-likelihood (MC) tasks.
  static v     A = mean_q Y[q,v],  C = mean_q c(q,v)
  hull         upper concave envelope of the static points in the (C, A) plane: what random
               mixing of two static variants achieves (RouterBench's baseline for routers)
  oracle       per query, the CHEAPEST variant that is correct; if none is correct, the cheapest
               variant overall (nothing can be gained). A = mean_q max_v Y[q,v]
  headroom     A_oracle - hull(C_oracle): accuracy the oracle gains over random mixing at the
               oracle's own cost. This is the P2 decision-gate quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def interp_prefill(lengths: list[int], ms: list[float], n: np.ndarray) -> np.ndarray:
    """Piecewise-linear prefill latency in prompt length, linear extrapolation at both ends."""
    x = np.asarray(lengths, dtype=float)
    y = np.asarray(ms, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = np.asarray(n, dtype=float)
    if len(x) == 1:
        return y[0] * n / x[0]
    out = np.interp(n, x, y)
    lo, hi = n < x[0], n > x[-1]
    out[lo] = y[0] + (n[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[hi] = y[-1] + (n[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.maximum(out, 0.0)


def query_costs(outcomes: pd.DataFrame, profiles: dict[str, dict]) -> pd.Series:
    """outcomes: rows (query_id, variant, n_ctx_tokens, n_gen_tokens); profiles[variant] has
    'prefill_lengths', 'prefill_ms', 'decode_ms_per_tok'. Returns cost in ms aligned to outcomes."""
    cost = pd.Series(np.nan, index=outcomes.index, dtype=float)
    for v, g in outcomes.groupby("variant"):
        p = profiles[v]
        pre = interp_prefill(p["prefill_lengths"], p["prefill_ms"], g["n_ctx_tokens"].to_numpy())
        cost.loc[g.index] = pre + g["n_gen_tokens"].to_numpy() * p["decode_ms_per_tok"]
    return cost


def wide(outcomes: pd.DataFrame, value: str) -> pd.DataFrame:
    w = outcomes.pivot(index="query_id", columns="variant", values=value)
    if w.isna().any().any():
        missing = int(w.isna().sum().sum())
        raise ValueError(f"outcome matrix incomplete: {missing} (query, variant) cells missing for '{value}'")
    return w


def upper_hull(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Upper concave envelope, non-decreasing in accuracy, of (cost, acc, name) points."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    hull: list[tuple[float, float, str]] = []
    for p in pts:
        if hull and p[1] <= hull[-1][1]:
            continue                                   # dominated: costs more, not more accurate
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()                             # middle point under the chord
            else:
                break
        hull.append(p)
    return hull


def hull_accuracy(hull: list[tuple[float, float, str]], cost: float) -> float:
    xs = [h[0] for h in hull]
    ys = [h[1] for h in hull]
    if cost <= xs[0]:
        return ys[0] if cost == xs[0] else float("nan")   # cheaper than any static point
    if cost >= xs[-1]:
        return ys[-1]
    return float(np.interp(cost, xs, ys))


def analyze(outcomes: pd.DataFrame, cost_col: str = "cost_ms", order: list[str] | None = None) -> dict:
    """outcomes: one row per (query_id, variant) with columns task, correct, and cost_col."""
    Y = wide(outcomes, "correct").astype(int)
    C = wide(outcomes, cost_col).astype(float)
    variants = order or list(Y.columns)
    Y, C = Y[variants], C[variants]
    task = outcomes.drop_duplicates("query_id").set_index("query_id")["task"].reindex(Y.index)

    static = {v: {"accuracy": float(Y[v].mean()), "cost_ms": float(C[v].mean())} for v in variants}
    per_task = {t: {v: float(Y.loc[task == t, v].mean()) for v in variants} for t in sorted(task.unique())}
    hull = upper_hull([(s["cost_ms"], s["accuracy"], v) for v, s in static.items()])

    # oracle: cheapest correct variant, else cheapest variant
    any_correct = Y.to_numpy().max(axis=1) == 1
    cost_if_correct = np.where(Y.to_numpy() == 1, C.to_numpy(), np.inf)
    pick_correct = np.argmin(cost_if_correct, axis=1)
    pick_cheapest = np.argmin(C.to_numpy(), axis=1)
    pick = np.where(any_correct, pick_correct, pick_cheapest)
    oracle_cost = float(C.to_numpy()[np.arange(len(pick)), pick].mean())
    oracle_acc = float(any_correct.mean())
    hull_at_oracle = hull_accuracy(hull, oracle_cost)

    # Per task: oracle vs. hull, and two references that separate real complementarity from noise:
    #  independent_oracle = 1 - prod_v (1 - A_v): oracle accuracy IF the variants' errors were independent
    #  chance = 1/k for k-option tasks: what pure guessing gets right
    n_choices = (outcomes.drop_duplicates("query_id").set_index("query_id")["n_choices"].reindex(Y.index)
                 if "n_choices" in outcomes.columns else None)
    per_task_headroom = {}
    for t in sorted(task.unique()):
        m = (task == t).to_numpy()
        Yt, Ct = Y.to_numpy()[m], C.to_numpy()[m]
        st = [(float(Ct[:, i].mean()), float(Yt[:, i].mean()), v) for i, v in enumerate(variants)]
        ht = upper_hull(st)
        anyc = Yt.max(axis=1) == 1
        pk = np.where(anyc, np.argmin(np.where(Yt == 1, Ct, np.inf), axis=1), np.argmin(Ct, axis=1))
        oc = float(Ct[np.arange(len(pk)), pk].mean())
        ha = hull_accuracy(ht, oc)
        accs = np.array([s[1] for s in st])
        k = None
        if n_choices is not None:
            kk = n_choices.to_numpy()[m]
            k = float(np.mean(kk[kk > 0])) if (kk > 0).any() else None
        per_task_headroom[t] = {
            "n": int(m.sum()), "oracle_accuracy": float(anyc.mean()), "oracle_cost_ms": oc,
            "hull_accuracy_at_oracle_cost": ha, "headroom_points": (float(anyc.mean()) - ha) if ha == ha else None,
            "best_static_accuracy": float(accs.max()), "independent_oracle_accuracy": float(1 - np.prod(1 - accs)),
            "chance_accuracy": (1.0 / k) if k else None,
        }

    ref = variants[0]
    pair = {}
    for v in variants[1:]:
        a, b = Y[v].to_numpy(), Y[ref].to_numpy()
        pair[v] = {"both_correct": float(((a == 1) & (b == 1)).mean()),
                   "only_pruned_correct": float(((a == 1) & (b == 0)).mean()),
                   "only_reference_correct": float(((a == 0) & (b == 1)).mean()),
                   "both_wrong": float(((a == 0) & (b == 0)).mean())}
    return {
        "n_queries": int(len(Y)), "variants": variants, "reference": ref,
        "static": static, "per_task_accuracy": per_task,
        "hull": [{"variant": h[2], "cost_ms": h[0], "accuracy": h[1]} for h in hull],
        "oracle": {"accuracy": oracle_acc, "cost_ms": oracle_cost,
                   "pick_share": {v: float((pick == i).mean()) for i, v in enumerate(variants)},
                   "unsolvable_share": float((~any_correct).mean())},
        "hull_accuracy_at_oracle_cost": hull_at_oracle,
        "headroom_points": (oracle_acc - hull_at_oracle) if hull_at_oracle == hull_at_oracle else None,
        "pairwise_vs_reference": pair,
        "per_task_headroom": per_task_headroom,
        "independent_oracle_accuracy": float(1 - np.prod([1 - s["accuracy"] for s in static.values()])),
    }
