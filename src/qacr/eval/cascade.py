"""Query-level confidence cascade: a cheap (pruned) variant answers first; its answer is kept only if its
confidence >= tau, otherwise the question is escalated to the full model.

Protocol (no tuning on the reported data):
  * the evaluated questions are split into two fixed halves by a hash of query_id (calibration / evaluation);
  * tau is chosen on the calibration half: the largest served fraction whose cascade accuracy is at most
    `max_loss` below the full model's accuracy on that half;
  * served fraction, accuracy and accuracy loss (paired bootstrap CI) are reported on the evaluation half.
Cost accounting: escalated questions pay for BOTH models, so with a per-question cost ratio r = c_small / c_full
the cascade is cheaper than the full model iff served_fraction > r (exactly, per question:
sum_q c_s(q) + sum_{escalated} c_f(q) < sum_q c_f(q)).
Generation questions (no answer options, e.g. GSM8K) have no option probabilities: they go straight to the
full model and are not part of the cascade population.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def option_confidence(lls, lm_choices, metric: str) -> tuple[float, float]:
    """(max probability, top1 - top2 margin) over answer options, under the task's primary normalisation."""
    x = np.asarray(lls, dtype=float)
    if metric == "acc_norm":
        x = x / np.asarray([len(c) for c in lm_choices], dtype=float)
    p = np.exp(x - x.max())
    p /= p.sum()
    s = np.sort(p)[::-1]
    return float(s[0]), float(s[0] - (s[1] if len(s) > 1 else 0.0))


def half_of(query_id: str) -> int:
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8], 16) % 2


def simulate(y_small: np.ndarray, y_full: np.ndarray, conf: np.ndarray, tau: float) -> dict:
    served = conf >= tau
    y = np.where(served, y_small, y_full)
    return {"served": float(served.mean()), "accuracy": float(y.mean()), "full_accuracy": float(y_full.mean()),
            "loss": float(y_full.mean() - y.mean())}


def choose_tau(y_small, y_full, conf, max_loss: float) -> float:
    """Largest served fraction with loss <= max_loss; candidate taus are the observed confidences (+ inf)."""
    best_tau, best_served = float("inf"), 0.0
    for tau in np.unique(conf):
        r = simulate(y_small, y_full, conf, tau)
        if r["loss"] <= max_loss + 1e-12 and r["served"] > best_served:
            best_tau, best_served = float(tau), r["served"]
    return best_tau


def paired_bootstrap_loss(y_small, y_full, served, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of (full accuracy - cascade accuracy) on the same questions."""
    rng = np.random.default_rng(seed)
    d = y_full - np.where(served, y_small, y_full)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_cascade(df: pd.DataFrame, small: str, full: str, max_loss: float, cost_col: str | None = None) -> dict:
    """df: rows (query_id, variant, correct, conf[, cost]) for MC questions only."""
    S = df[df["variant"] == small].set_index("query_id").sort_index()
    F = df[df["variant"] == full].set_index("query_id").sort_index()
    if not S.index.equals(F.index):
        raise ValueError("small and full variants cover different questions")
    halves = np.array([half_of(q) for q in S.index])
    out = {"small": small, "full": full, "max_loss": max_loss, "n_calibration": int((halves == 0).sum()),
           "n_evaluation": int((halves == 1).sum())}
    cal, ev = halves == 0, halves == 1
    ys, yf, c = S["correct"].to_numpy(float), F["correct"].to_numpy(float), S["conf"].to_numpy(float)
    tau = choose_tau(ys[cal], yf[cal], c[cal], max_loss)
    out["tau"] = tau
    out["calibration"] = simulate(ys[cal], yf[cal], c[cal], tau)
    r = simulate(ys[ev], yf[ev], c[ev], tau)
    served = c[ev] >= tau
    r["loss_ci95"] = paired_bootstrap_loss(ys[ev], yf[ev], served)
    r["small_accuracy_on_served"] = float(ys[ev][served].mean()) if served.any() else None
    r["full_accuracy_on_served"] = float(yf[ev][served].mean()) if served.any() else None
    out["evaluation"] = r
    if cost_col is not None:
        cs, cf = S[cost_col].to_numpy(float)[ev], F[cost_col].to_numpy(float)[ev]
        cascade_cost = cs.sum() + cf[~served].sum()
        out["evaluation"]["cost_ratio_small_over_full"] = float(cs.sum() / cf.sum())
        out["evaluation"]["cascade_cost_over_full"] = float(cascade_cost / cf.sum())
    # curve over fixed taus on the evaluation half (for the figure)
    out["curve"] = [{"tau": float(t), **simulate(ys[ev], yf[ev], c[ev], float(t))}
                    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)]
    return out


def break_even(served: float, cost_ratio: float) -> dict:
    """Cascade relative cost = r + (1 - served); cheaper than the full model iff served > r."""
    rel = cost_ratio + (1.0 - served)
    return {"relative_cost": rel, "saves": bool(rel < 1.0), "required_served_fraction": cost_ratio}
