import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qacr.router.estimators import KNN, MLP, ExternalDifficulty, LogReg, Platt, RuleTable, TaskTable, reliability
from qacr.router.features import Features, HashingEncoder, handcrafted, query_text
from qacr.router.policy import (Router, area_above_hull, bootstrap, curve, expected_gen_tokens, mcnemar, predicted_cost,
                                replay, route, select, static_points)
from qacr.eval.headroom import upper_hull

REPO = Path(__file__).resolve().parents[1]


def test_query_text_uses_only_the_query():
    pool = pd.DataFrame({"context": [None, "passage"], "question_text": ["q1", "q2"],
                         "choices": [np.array(["a", "b"], dtype=object), None]})
    assert query_text(pool) == ["q1 a b", "passage q2"]


def test_handcrafted_counts():
    h = handcrafted(["what is 12 + 3", "plain words"], np.array([0, 4]))
    assert h.shape == (2, 6)
    assert h[0, 2] == pytest.approx(np.log1p(3)) and h[0, 3] == pytest.approx(np.log1p(1))
    assert h[1, 2] == 0 and h[1, 4] == 1 and h[1, 5] == 4 and h[0, 4] == 0


def _features(n=400, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    hand = np.column_stack([rng.normal(size=(n, 2)), np.abs(rng.normal(size=n)), np.zeros(n), np.ones(n), 4 * np.ones(n)])
    task = np.array(["a", "b"])[rng.integers(0, 2, n)]
    # variant 0 is right when the first embedding dimension is positive; variant 1 on task "a" only
    Y = np.column_stack([emb[:, 0] > 0, task == "a"]).astype(float)
    return Features(emb, hand, task), Y


@pytest.mark.parametrize("est", [LogReg(), KNN(k=15), MLP(hidden=16, max_iter=300)])
def test_content_estimators_learn_the_content_signal(est):
    f, Y = _features(3000)
    fv, Yv = _features(seed=1)
    P = est.fit(f, Y).predict(fv)
    assert P.shape == Yv.shape and P.min() >= 0 and P.max() <= 1
    assert ((P[:, 0] > 0.5) == (Yv[:, 0] > 0)).mean() > 0.8


def test_task_table_reads_task_and_rule_table_does_not():
    f, Y = _features()
    fv, _ = _features(seed=1)
    P = TaskTable().fit(f, Y).predict(fv)
    assert P[fv.task == "a", 1].min() > 0.9 and P[fv.task == "b", 1].max() < 0.1
    rule = RuleTable().fit(f, Y)
    shuffled = Features(fv.embedding, fv.hand, np.array(["zzz"] * len(fv)))
    assert np.array_equal(rule.predict(fv), rule.predict(shuffled))


def test_single_class_variant_does_not_break_fitting():
    f, Y = _features()
    Y[:, 1] = 1.0
    P = LogReg().fit(f, Y).predict(f)
    assert (P[:, 1] == 1.0).all()


def test_external_difficulty_ignores_unlabelled_queries_in_regression():
    f, Y = _features()
    d = f.embedding[:, 0].copy()
    d[::2] = np.nan
    P = ExternalDifficulty().fit(f, Y, d).predict(f)
    assert P.shape == Y.shape and ((P[:, 0] > 0.5) == (Y[:, 0] > 0)).mean() > 0.8


def test_platt_fixes_a_shifted_estimator():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=5000) < p).astype(float)
    biased = np.clip(p + 0.2, 0, 1)[:, None]
    fixed = Platt().fit(biased, y[:, None]).predict(biased)
    assert reliability(fixed[:, 0], y)["ece"] < reliability(biased[:, 0], y)["ece"]


def test_reliability_perfect_and_constant():
    r = reliability(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert r["brier"] == 0 and r["ece"] == 0
    assert sum(b["n"] for b in reliability(np.full(10, 0.5), np.ones(10))["table"]) == 10


def test_route_follows_lambda_and_breaks_ties_to_cheaper():
    P = np.array([[0.9, 0.7], [0.5, 0.5]])
    C = np.array([[1.0, 0.5], [1.0, 0.5]])
    assert route(P, C, 0.0).tolist() == [0, 1]
    assert route(P, C, 1.0).tolist() == [1, 1]


def test_router_callable_returns_variant_names():
    f, Y = _features(50)
    r = Router(TaskTable().fit(f, Y), ["full", "cheap"], lambda fx: np.tile([1.0, 0.1], (len(fx), 1)))
    assert set(r.route(f, 100.0)) == {"cheap"}


def test_predicted_cost_is_free_of_realised_outcomes():
    prof = {"a": {"prefill_lengths": [10, 20], "prefill_ms": [1.0, 2.0], "decode_ms_per_tok": 1.0},
            "b": {"prefill_lengths": [10, 20], "prefill_ms": [1.0, 2.0], "decode_ms_per_tok": 0.5}}
    is_mcq = np.array([True, False])
    gen = expected_gen_tokens(np.array([[0, 0], [8, 12]]), is_mcq)
    assert gen.tolist() == [8, 12]
    C, unit = predicted_cost(np.array([10, 10]), is_mcq, gen, ["a", "b"], prof)
    assert unit == pytest.approx((1 + 9) / 2) and C[1, 1] == pytest.approx((1 + 6) / unit)
    assert C[0, 0] == C[0, 1]                                   # MCQ: no generation term


def test_select_targets():
    c = pd.DataFrame({"lambda": [0.0, 1.0, 2.0], "accuracy": [0.80, 0.795, 0.70], "cost_ms": [10.0, 8.0, 5.0]})
    assert select(c, "best_accuracy", 0.8) == 0.0
    assert select(c, "iso_1pt", 0.8) == 1.0
    assert select(c, "iso_95pct", 0.8) == 1.0
    assert select(c[c["lambda"] == 2.0], "iso_1pt", 0.8) is None


def test_replay_charges_overhead_and_area_above_hull():
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    C = np.array([[10, 4]] * 4, dtype=float)
    r = replay(np.array([0, 1, 1, 1]), Y, C, np.array(["t"] * 4), overhead_ms=1.0)
    assert r["accuracy"] == 0.75 and r["cost_ms"] == pytest.approx(5.5 + 1.0)
    hull = upper_hull(static_points(Y, C, ["a", "b"]))
    assert len(hull) == 1                                       # equal accuracy: the cheaper one dominates
    hull = [(4.0, 0.25, "b"), (10.0, 0.75, "a")]
    c = pd.DataFrame({"cost_ms": [4.0, 10.0], "accuracy": [0.5, 0.75]})
    assert area_above_hull(c, hull) == pytest.approx(0.125, abs=2e-3)
    assert area_above_hull(pd.DataFrame({"cost_ms": [11.0], "accuracy": [0.9]}), hull) is None


def test_mcnemar_and_bootstrap_on_identical_router():
    y = np.array([1, 0] * 50)
    assert mcnemar(y, y) == {"only_a": 0, "only_b": 0, "p": 1.0}
    Y = np.column_stack([y, 1 - y]).astype(float)
    C = np.tile([10.0, 5.0], (100, 1))
    b = bootstrap(np.zeros(100, int), Y, C, 0.0, ["full", "cheap"], n=200)
    assert b["accuracy_vs_unpruned_ci95"] == [0.0, 0.0] and b["cost_saving_vs_unpruned_ci95"] == [0.0, 0.0]


def test_curve_has_one_row_per_lambda():
    P = np.array([[0.9, 0.7], [0.5, 0.5]])
    C = np.array([[1.0, 0.5], [1.0, 0.5]])
    Y = np.array([[1.0, 0.0], [1.0, 1.0]])
    c = curve(P, C, [0.0, 1.0], Y, C * 10, np.array(["t", "t"]))
    assert c["accuracy"].tolist() == [1.0, 0.5] and c["cost_ms"].tolist() == [7.5, 5.0]


# --- end to end on synthetic files ------------------------------------------------------------------------------

def _load_script():
    spec = importlib.util.spec_from_file_location("p3_routers", REPO / "scripts" / "p3_routers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(tmp_path, n=600, test_correct_flip=False):
    rng = np.random.default_rng(0)
    split = np.array(["train", "val", "test"])[rng.choice(3, n, p=[.6, .2, .2])]
    easy = rng.uniform(size=n) < 0.5
    pool = pd.DataFrame({
        "query_id": [f"q{i:04d}" for i in range(n)],
        "task": np.array(["t1", "t2"])[rng.integers(0, 2, n)],
        "question_text": [("easy cat dog" if e else "hard 12 + 3 = what") + f" {i}" for i, e in enumerate(easy)],
        "context": [None] * n, "choices": [np.array(["a", "b"], dtype=object)] * n, "n_choices": 2, "split": split})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)
    rb = pd.DataFrame({"query_id": pool["query_id"][:200], "rb_mean_score": easy[:200].astype(float)})
    rb_path = tmp_path / "rb.parquet"
    rb.to_parquet(rb_path)
    dirs = {"trainval": {}, "test": {}}
    params = {"full": 100, "cheap": 50}
    for v, p in (("full", 0.85), ("cheap", None)):
        correct = (rng.uniform(size=n) < p).astype(int) if p else (easy & (rng.uniform(size=n) < 0.9)).astype(int)
        for part, mask in (("trainval", split != "test"), ("test", split == "test")):
            c = 1 - correct if (test_correct_flip and part == "test") else correct
            d = tmp_path / f"{part}_{v}"
            d.mkdir()
            pd.DataFrame({"query_id": pool["query_id"][mask], "variant": v, "task": pool["task"][mask], "correct": c[mask],
                          "n_ctx_tokens": 50, "n_gen_tokens": 0, "prompt_hash": "h", "gold": "0", "pred": 0}
                         ).to_parquet(d / "outcomes.parquet")
            (d / "variant.json").write_text(json.dumps({"params": {"total": params[v]}}))
            dirs[part][v] = str(d)
    prof = tmp_path / "prof"
    prof.mkdir()
    (prof / "profiles.json").write_text(json.dumps({
        "full": {"prefill": {"128": {"median_ms": 20.0}, "512": {"median_ms": 80.0}}, "generate": {"decode_ms_per_tok": 30.0}},
        "cheap": {"prefill": {"128": {"median_ms": 10.0}, "512": {"median_ms": 40.0}}, "generate": {"decode_ms_per_tok": 15.0}}}))
    cfg = {"phase": "p3", "run_name": "routers", "seed": 0, "bootstrap": 50, "evaluate_test": True,
           "paths": {"results_root": str(tmp_path / "out")}, "pool": {"path": str(pool_path)},
           "routerbench_labels": str(rb_path), "encoder": {"kind": "hashing"}, "outcomes": dirs,
           "variant_profile_names": {"full": "full", "cheap": "cheap"},
           "profiles": {"dev": {"dir": str(prof)}}, "param_ratio": {"base": "dev"},
           "routers": {"logreg_C": [1.0], "knn_k": [20], "mlp_hidden": 16, "mlp_alpha": [0.001], "mlp_seeds": [0, 1]}}
    import yaml
    path = tmp_path / f"cfg_{test_correct_flip}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def _run(mod, path, cfg):
    assert mod.main(["--config", str(path)]) == 0
    runs = sorted((Path(cfg["paths"]["results_root"]) / "p3").iterdir(), key=lambda p: p.stat().st_mtime_ns)
    return runs[-1]


def test_end_to_end_routes_easy_queries_down_and_beats_hull(tmp_path):
    mod = _load_script()
    path, cfg = _world(tmp_path)
    out = _run(mod, path, cfg)
    for f in ("summary.csv", "selection.json", "calibration.json", "overhead_ms.json", "curves_dev.csv", "results_dev.json",
              "results_param_ratio.json", "splits_accuracy.csv", "encoder.json"):
        assert (out / f).exists(), f
    tab = pd.read_csv(out / "summary.csv")
    assert set(tab["cost_model"]) == {"dev", "param_ratio"}
    assert {"logreg", "mlp", "knn", "task_table", "rule_table", "external_difficulty"} <= set(tab["router"])
    row = tab[(tab.cost_model == "param_ratio") & (tab.router == "logreg") & (tab.target == "iso_95pct")].iloc[0]
    assert row["saving_vs_unpruned"] > 0.05 and row["gain_over_hull_points"] > 0     # the planted signal is found
    res = json.loads((out / "results_dev.json").read_text())
    assert len(res["routers"]["mlp"]["seeds"]) == 2
    assert res["oracle_test"]["oracle"]["accuracy"] >= res["static_test"]["full"]["accuracy"]


def test_operating_points_do_not_depend_on_test_labels(tmp_path):
    mod = _load_script()
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    pa, ca = _world(a)
    pb, cb = _world(b, test_correct_flip=True)
    sa = json.loads((_run(mod, pa, ca) / "selection.json").read_text())
    sb = json.loads((_run(mod, pb, cb) / "selection.json").read_text())
    assert sa == sb


def test_evaluate_test_false_needs_no_test_outcomes(tmp_path):
    mod = _load_script()
    path, cfg = _world(tmp_path)
    cfg["outcomes"]["test"] = {"full": "does/not/exist", "cheap": "does/not/exist"}
    cfg["evaluate_test"] = False
    import yaml
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = _run(mod, path, cfg)
    assert not (out / "summary.csv").exists() and (out / "selection.json").exists()
    assert "static_test" not in json.loads((out / "results_dev.json").read_text())


def test_incomplete_outcomes_are_rejected(tmp_path):
    mod = _load_script()
    path, cfg = _world(tmp_path)
    d = Path(cfg["outcomes"]["trainval"]["cheap"]) / "outcomes.parquet"
    df = pd.read_parquet(d)
    df.iloc[:-1].to_parquet(d)
    with pytest.raises(ValueError, match="does not cover"):
        mod.main(["--config", str(path)])


def test_hashing_encoder_is_deterministic():
    e = HashingEncoder(16)
    assert np.array_equal(e.encode(["a b"]), e.encode(["a b"], device="cpu"))
