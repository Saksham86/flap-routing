import numpy as np
import pandas as pd
import pytest

from qacr.eval.headroom import analyze, hull_accuracy, interp_prefill, query_costs, upper_hull


def test_interp_prefill_inside_and_extrapolated():
    f = interp_prefill([128, 512], [10.0, 40.0], np.array([128, 320, 512, 1024, 64]))
    assert np.allclose(f, [10.0, 25.0, 40.0, 80.0, 5.0])


def test_upper_hull_drops_dominated_and_concave_violations():
    pts = [(1.0, 0.5, "a"), (2.0, 0.55, "b"), (3.0, 0.7, "c"), (2.5, 0.4, "d")]
    h = upper_hull(pts)
    assert [p[2] for p in h] == ["a", "c"]          # b lies under chord a-c; d dominated
    assert hull_accuracy(h, 2.0) == pytest.approx(0.6)
    assert hull_accuracy(h, 5.0) == 0.7
    assert np.isnan(hull_accuracy(h, 0.5))


def _toy():
    # 4 queries, variants v0 (expensive) and v1 (cheap)
    rows = []
    Y = {"v0": [1, 1, 0, 0], "v1": [1, 0, 1, 0]}
    cost = {"v0": 10.0, "v1": 4.0}
    for v in Y:
        for i in range(4):
            rows.append({"query_id": f"q{i}", "variant": v, "task": "t", "correct": Y[v][i], "cost_ms": cost[v]})
    return pd.DataFrame(rows)


def test_analyze_oracle_and_headroom():
    r = analyze(_toy(), order=["v0", "v1"])
    assert r["static"]["v0"]["accuracy"] == 0.5 and r["static"]["v1"]["accuracy"] == 0.5
    # oracle: q0 -> v1 (cheapest correct), q1 -> v0, q2 -> v1, q3 unsolvable -> v1 (cheapest)
    assert r["oracle"]["accuracy"] == 0.75
    assert r["oracle"]["cost_ms"] == pytest.approx((4 + 10 + 4 + 4) / 4)
    assert r["oracle"]["unsolvable_share"] == 0.25
    # hull: v1 dominates v0 (same accuracy, cheaper) -> hull = [v1]; oracle cost 5.5 > 4 -> 0.5
    assert r["hull_accuracy_at_oracle_cost"] == 0.5 and r["headroom_points"] == pytest.approx(0.25)
    p = r["pairwise_vs_reference"]["v1"]
    assert p == {"both_correct": 0.25, "only_pruned_correct": 0.25, "only_reference_correct": 0.25, "both_wrong": 0.25}


def test_incomplete_matrix_raises():
    df = _toy().iloc[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        analyze(df)


def test_query_costs_uses_ctx_and_gen():
    out = pd.DataFrame({"query_id": ["a", "b"], "variant": ["v", "v"], "n_ctx_tokens": [128, 512],
                        "n_gen_tokens": [0, 10]})
    prof = {"v": {"prefill_lengths": [128, 512], "prefill_ms": [10.0, 40.0], "decode_ms_per_tok": 2.0}}
    assert list(query_costs(out, prof)) == [10.0, 60.0]
