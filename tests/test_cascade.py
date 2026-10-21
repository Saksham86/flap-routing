import numpy as np
import pandas as pd
import pytest

from qacr.eval.cascade import break_even, choose_tau, evaluate_cascade, half_of, option_confidence, simulate


def test_option_confidence_norms():
    p, m = option_confidence([-1.0, -1.0], ["a", "bb"], "acc")
    assert p == pytest.approx(0.5) and m == pytest.approx(0.0)
    p, _ = option_confidence([-1.0, -2.0], ["a", "bbbb"], "acc_norm")     # -1/1 vs -2/4=-0.5 -> option 2 wins
    assert p > 0.5


def test_simulate_and_choose_tau():
    ys = np.array([1, 1, 0, 0])
    yf = np.array([1, 1, 1, 0])
    conf = np.array([0.9, 0.8, 0.95, 0.1])
    r = simulate(ys, yf, conf, 0.85)          # serves q0 (right) and q2 (wrong)
    assert r["served"] == 0.5 and r["loss"] == pytest.approx(0.25)
    tau = choose_tau(ys, yf, conf, max_loss=0.0)
    # the most confident question (q2, 0.95) is one the small model gets wrong, so every threshold that serves
    # anything also serves q2 and loses accuracy: the only lossless choice is to serve nothing
    assert simulate(ys, yf, conf, tau) == {"served": 0.0, "accuracy": 0.75, "full_accuracy": 0.75, "loss": 0.0}
    assert simulate(ys, yf, conf, choose_tau(ys, yf, conf, max_loss=0.25))["served"] == 1.0   # loss 0.25 allowed -> serve all


def test_threshold_chosen_without_evaluation_half():
    rows = []
    for i in range(200):
        q = f"q{i:04d}"
        rows.append({"query_id": q, "variant": "full", "correct": 1, "conf": 1.0, "cost": 10.0})
        rows.append({"query_id": q, "variant": "small", "correct": int(i % 3 != 0), "conf": 0.9 if i % 3 else 0.6,
                     "cost": 4.0})
    out = evaluate_cascade(pd.DataFrame(rows), "small", "full", max_loss=0.0, cost_col="cost")
    assert out["n_calibration"] + out["n_evaluation"] == 200
    ev = out["evaluation"]
    assert ev["loss"] == pytest.approx(0.0) and ev["served"] > 0.5      # confident answers are the right ones
    assert ev["cost_ratio_small_over_full"] == pytest.approx(0.4)
    assert ev["cascade_cost_over_full"] == pytest.approx(0.4 + (1 - ev["served"]))


def test_break_even_rule():
    assert break_even(0.3, 0.8)["saves"] is False
    assert break_even(0.7, 0.5)["saves"] is True
    assert half_of("abc") in (0, 1) and half_of("abc") == half_of("abc")
