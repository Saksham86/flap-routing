"""Routing rule and offline replay.

  pi_lambda(q) = argmax_v  p_hat[q, v] - lambda * c_tilde[q, v]

c_tilde is what the router can know BEFORE running anything: measured prefill at the query's prompt length, plus
the expected generated tokens of that variant (train mean) times its decode cost, in units of the unpruned model's
mean cost. Replay then charges the cost actually incurred, plus the router's own overhead.

Operating points are picked on val and only then replayed on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from qacr.eval.headroom import hull_accuracy, interp_prefill, upper_hull

TARGETS = ("iso_1pt", "iso_95pct", "best_accuracy")


def expected_gen_tokens(n_gen: np.ndarray, is_mcq: np.ndarray) -> np.ndarray:
    """Train-split mean tokens per variant for generation queries: (n_variants,). MCQ queries generate none."""
    return n_gen[~is_mcq].mean(axis=0) if (~is_mcq).any() else np.zeros(n_gen.shape[1])


def predicted_cost(n_ctx: np.ndarray, is_mcq: np.ndarray, gen_mean: np.ndarray, variants: list[str],
                   profiles: dict[str, dict], unit: float | None = None) -> tuple[np.ndarray, float]:
    """c_tilde and its unit (the unpruned model's mean predicted cost; pass the train value for val/test)."""
    C = np.zeros((len(n_ctx), len(variants)))
    for j, v in enumerate(variants):
        p = profiles[v]
        C[:, j] = interp_prefill(p["prefill_lengths"], p["prefill_ms"], n_ctx) + (~is_mcq) * gen_mean[j] * p["decode_ms_per_tok"]
    unit = float(C[:, 0].mean()) if unit is None else unit
    return C / unit, unit


def route(P: np.ndarray, C_tilde: np.ndarray, lam: float) -> np.ndarray:
    """Index of the chosen variant per query; ties go to the cheaper predicted cost."""
    score = P - lam * C_tilde
    best = score.max(axis=1, keepdims=True)
    return np.where(np.isclose(score, best, rtol=0, atol=1e-12), C_tilde, np.inf).argmin(axis=1)


class Router:
    """route(features, lambda) -> variant names: the callable that Phase 4's pipeline uses."""

    def __init__(self, estimator, variants: list[str], cost_fn, platt=None):
        self.estimator, self.variants, self.cost_fn, self.platt = estimator, variants, cost_fn, platt

    def route(self, features, lam: float, *cost_args) -> list[str]:
        P = self.estimator.predict(features)
        if self.platt is not None:
            P = self.platt.predict(P)
        return [self.variants[i] for i in route(P, self.cost_fn(features, *cost_args), lam)]


def replay(pick: np.ndarray, Y: np.ndarray, C: np.ndarray, task: np.ndarray | None = None, overhead_ms: float = 0.0) -> dict:
    rows = np.arange(len(pick))
    y = Y[rows, pick]
    out = {"accuracy": float(y.mean()), "cost_ms": float(C[rows, pick].mean() + overhead_ms),
           "share": np.bincount(pick, minlength=Y.shape[1]).astype(float) / len(pick)}
    if task is not None:
        out["macro_accuracy"] = float(pd.Series(y).groupby(task).mean().mean())
    return out


def curve(P: np.ndarray, C_tilde: np.ndarray, lambdas: list[float], Y: np.ndarray, C: np.ndarray, task: np.ndarray,
          overhead_ms: float = 0.0) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        r = replay(route(P, C_tilde, lam), Y, C, task, overhead_ms)
        share = r.pop("share")
        rows.append({"lambda": float(lam), **r, **{f"share_{j}": float(s) for j, s in enumerate(share)}})
    return pd.DataFrame(rows)


def select(val_curve: pd.DataFrame, target: str, reference_accuracy: float) -> float | None:
    """lambda for a target, read off the VAL curve. None if the target is unreachable on val.
    iso_1pt: cheapest with accuracy >= reference - 1 pt; iso_95pct: cheapest with accuracy >= 95% of reference;
    best_accuracy: highest accuracy, then cheapest."""
    if target == "best_accuracy":
        c = val_curve.sort_values(["accuracy", "cost_ms", "lambda"], ascending=[False, True, False])
    else:
        floor = {"iso_1pt": reference_accuracy - 0.01, "iso_95pct": 0.95 * reference_accuracy}[target]
        c = val_curve[val_curve["accuracy"] >= floor].sort_values(["cost_ms", "accuracy", "lambda"],
                                                                  ascending=[True, False, False])
    return None if c.empty else float(c.iloc[0]["lambda"])


def static_points(Y: np.ndarray, C: np.ndarray, variants: list[str]) -> list[tuple[float, float, str]]:
    return [(float(C[:, j].mean()), float(Y[:, j].mean()), v) for j, v in enumerate(variants)]


def area_above_hull(c: pd.DataFrame, hull: list[tuple[float, float, str]], grid: int = 200) -> float | None:
    """Mean accuracy gain of the router curve's upper envelope over the hull, across the cost range they share.
    None if they share no cost range."""
    pts = upper_hull([(float(a), float(b), "") for a, b in zip(c["cost_ms"], c["accuracy"])])
    lo, hi = max(pts[0][0], hull[0][0]), min(pts[-1][0], hull[-1][0])
    if len(pts) < 2 or hi <= lo:
        return None
    x = np.linspace(lo, hi, grid)
    return float((np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
                  - np.interp(x, [h[0] for h in hull], [h[1] for h in hull])).mean())


def mcnemar(y_a: np.ndarray, y_b: np.ndarray) -> dict:
    only_a, only_b = int(((y_a == 1) & (y_b == 0)).sum()), int(((y_a == 0) & (y_b == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}


def _ci(x: np.ndarray) -> list[float] | None:
    x = x[~np.isnan(x)]
    return None if not len(x) else [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def bootstrap(pick: np.ndarray, Y: np.ndarray, C: np.ndarray, overhead_ms: float, variants: list[str],
              n: int = 10000, seed: int = 0, chunk: int = 250) -> dict:
    """Paired bootstrap over queries against the unpruned model (column 0) and the hull of fixed variants.
    The hull is rebuilt inside each resample. A resample in which the router is cheaper than every fixed variant
    has no hull value there; those are excluded from the hull CI and counted."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(pick))
    y, c = Y[rows, pick].astype(float), C[rows, pick] + overhead_ms
    d_acc, saving, gain = [], [], []
    for start in range(0, n, chunk):
        idx = rng.integers(0, len(pick), size=(min(chunk, n - start), len(pick)))
        acc, cost = y[idx].mean(axis=1), c[idx].mean(axis=1)
        A, K = Y[idx].mean(axis=1), C[idx].mean(axis=1)
        d_acc.append(acc - A[:, 0])
        saving.append(1 - cost / K[:, 0])
        for b in range(len(idx)):
            hull = upper_hull([(float(K[b, j]), float(A[b, j]), v) for j, v in enumerate(variants)])
            gain.append(acc[b] - hull_accuracy(hull, float(cost[b])))
    gain = np.asarray(gain)
    return {"accuracy_vs_unpruned_ci95": _ci(np.concatenate(d_acc)), "cost_saving_vs_unpruned_ci95": _ci(np.concatenate(saving)),
            "gain_over_hull_ci95": _ci(gain), "resamples": n, "resamples_below_hull_range": int(np.isnan(gain).sum())}
