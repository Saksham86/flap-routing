"""P3 (CPU or GPU for the embeddings): train routers on train, pick operating points on val, replay once on test.

Inputs (config):
  outcomes.trainval / outcomes.test: {variant: <p2 outcome run dir>}; the first variant is the unpruned reference
  profiles: {label: {dir, per_variant_dirs}} measured cost models; param_ratio: a labelled reference, not a measurement
Routers: fixed variants, random mix, uniform random, task table (reads the task id), rule table, logistic regression,
  MLP, kNN, external difficulty (RouterBench), per-query and informative oracles.
Outputs in results/p3/<run_id>/: curves_<label>.csv, results_<label>.json, summary.csv, calibration.json,
  splits_accuracy.csv, selection.json. With evaluate_test=false nothing is read from the test split.

Usage: python scripts/p3_routers.py --config configs/p3_routers.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.eval.headroom import analyze, hull_accuracy, load_profiles, query_costs, upper_hull, wide  # noqa: E402
from qacr.router.estimators import KNN, MLP, ExternalDifficulty, LogReg, Platt, RuleTable, TaskTable, reliability  # noqa: E402
from qacr.router.features import Features, handcrafted, make_encoder, query_text  # noqa: E402
from qacr.router.policy import (TARGETS, area_above_hull, bootstrap, curve, expected_gen_tokens, mcnemar,  # noqa: E402
                                predicted_cost, replay, route, select, static_points)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and math.isnan(o) else o


def load_long(dirs: dict[str, str], pool: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    """Outcomes of every variant on exactly the pool's queries of these splits, with identical prompts."""
    want = pool.loc[pool["split"].isin(splits), "query_id"]
    frames = []
    for v, d in dirs.items():
        df = pd.read_parquet(Path(d) / "outcomes.parquet")
        if set(df["variant"].unique()) != {v}:
            raise ValueError(f"{d}: contains variants {set(df['variant'].unique())}, expected {v}")
        df = df[df["query_id"].isin(set(want))]
        if set(df["query_id"]) != set(want) or df["query_id"].duplicated().any():
            raise ValueError(f"{d}: does not cover splits {splits} exactly ({df['query_id'].nunique()} of {len(want)} queries)")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    bad = out.groupby("query_id")["prompt_hash"].nunique() > 1
    if bad.any():
        raise ValueError(f"{int(bad.sum())} queries have different prompts across variants")
    return out.merge(pool[["query_id", "n_choices"]], on="query_id", how="left", validate="many_to_one")


def param_ratio_profiles(base: dict[str, dict], dirs: dict[str, str], variants: list[str]) -> dict[str, dict]:
    """The unpruned model's measured profile scaled by each variant's share of parameters: a device-independent
    reference for 'cost follows parameter count'. Not a measurement."""
    total = {v: json.loads((Path(d) / "variant.json").read_text(encoding="utf-8"))["params"]["total"] for v, d in dirs.items()}
    b = base[variants[0]]
    return {v: {"prefill_lengths": b["prefill_lengths"], "prefill_ms": [m * total[v] / total[variants[0]] for m in b["prefill_ms"]],
                "decode_ms_per_tok": b["decode_ms_per_tok"] * total[v] / total[variants[0]]} for v in variants}


def embed_all(encoder, ids: np.ndarray, texts: list[str], cache: Path | None, log) -> np.ndarray:
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["encoder"]) == f"{encoder.name}@{encoder.commit}" and np.array_equal(z["ids"], ids.astype(str)):
            log.info("embeddings from cache %s", cache)
            return z["embedding"]
    t0 = time.perf_counter()
    emb = encoder.encode(texts)
    log.info("embedded %d queries in %.1fs", len(texts), time.perf_counter() - t0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=ids.astype(str), embedding=emb, encoder=f"{encoder.name}@{encoder.commit}")
    return emb


def fit_candidates(make, grid: list, f_tr: Features, Y_tr: np.ndarray, f_val: Features, Y_val: np.ndarray, *fit_args):
    """Fit one estimator per hyper-parameter, keep the lowest val Brier score."""
    best = None
    for h in grid:
        est = make(h).fit(f_tr, Y_tr, *fit_args)
        brier = float(((est.predict(f_val) - Y_val) ** 2).mean())
        if best is None or brier < best[2]:
            best = (est, h, brier)
    return best


def measure_overhead(fn, n_rep: int = 30) -> float:
    fn()                                  # warm-up
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p3.routers", run.path("run.log"))

    pool = pd.read_parquet(REPO / get_dotted(cfg, "pool.path", "data/p2/query_pool.parquet")).sort_values("query_id")
    evaluate_test = bool(cfg.get("evaluate_test", True))
    variants = list(cfg["outcomes"]["trainval"].keys())
    if evaluate_test and list(cfg["outcomes"]["test"].keys()) != variants:
        raise ValueError("outcomes.trainval and outcomes.test must list the same variants in the same order")

    long_tv = load_long(cfg["outcomes"]["trainval"], pool, ["train", "val"])
    long_te = load_long(cfg["outcomes"]["test"], pool, ["test"]) if evaluate_test else None
    log.info("variants %s; train+val rows %d; test rows %s", variants, len(long_tv), None if long_te is None else len(long_te))

    used = pool[pool["split"].isin(["train", "val", "test"] if evaluate_test else ["train", "val"])]
    ids = used["query_id"].to_numpy()
    texts = query_text(used)
    encoder = make_encoder(cfg.get("encoder", {}))
    run.write_json("encoder.json", {"name": encoder.name, "commit": encoder.commit})
    cache = cfg.get("embedding_cache")
    emb = embed_all(encoder, ids, texts, Path(cache) if cache else None, log)
    hand = handcrafted(texts, used["n_choices"].to_numpy())
    feats = Features(emb, hand, used["task"].astype(str).to_numpy())
    split_of = used["split"].to_numpy()
    fx = {s: feats.take(np.flatnonzero(split_of == s)) for s in np.unique(split_of)}
    qid = {s: ids[split_of == s] for s in fx}
    is_mcq = {s: (used["n_choices"].to_numpy()[split_of == s] > 0) for s in fx}

    def matrix(long: pd.DataFrame, s: str, col: str) -> np.ndarray:
        return wide(long[long["query_id"].isin(set(qid[s]))], col).loc[qid[s], variants].to_numpy(dtype=float)

    long_of = {"train": long_tv, "val": long_tv, "test": long_te}
    Y = {s: matrix(long_of[s], s, "correct") for s in fx}
    n_gen = {s: matrix(long_of[s], s, "n_gen_tokens") for s in fx}
    n_ctx = {s: matrix(long_of[s], s, "n_ctx_tokens")[:, 0] for s in fx}

    # Do the pruned variants look better on the questions FLAP was calibrated on?
    acc_tab = pd.DataFrame({s: dict(zip(variants, Y[s].mean(axis=0))) for s in fx})
    acc_tab.to_csv(run.path("splits_accuracy.csv"))
    log.info("accuracy per split:\n%s", acc_tab.round(4).to_string())

    # --- estimators: fit on train, hyper-parameters and Platt scaling on val --------------------------------------
    rb = pd.read_parquet(REPO / get_dotted(cfg, "routerbench_labels", "data/p2/routerbench_labels.parquet")).set_index("query_id")
    d_train = rb["rb_mean_score"].reindex(qid["train"]).to_numpy(dtype=float)
    r = cfg.get("routers", {})
    seeds = list(r.get("mlp_seeds", [0, 1, 2, 3, 4]))
    fitted: dict[str, list] = {}
    chosen = {}
    fitted["task_table"] = [TaskTable().fit(fx["train"], Y["train"])]
    fitted["rule_table"] = [RuleTable().fit(fx["train"], Y["train"])]
    est, h, _ = fit_candidates(lambda C: LogReg(C=C), r.get("logreg_C", [0.01, 0.1, 1.0]), fx["train"], Y["train"], fx["val"], Y["val"])
    fitted["logreg"], chosen["logreg_C"] = [est], h
    est, h, _ = fit_candidates(lambda k: KNN(k=k), r.get("knn_k", [10, 20, 50, 100]), fx["train"], Y["train"], fx["val"], Y["val"])
    fitted["knn"], chosen["knn_k"] = [est], h
    _, h, _ = fit_candidates(lambda a: MLP(hidden=int(r.get("mlp_hidden", 128)), alpha=a, seed=seeds[0]), r.get("mlp_alpha", [1e-3, 1e-1]),
                             fx["train"], Y["train"], fx["val"], Y["val"])
    chosen["mlp_alpha"] = h
    fitted["mlp"] = [MLP(hidden=int(r.get("mlp_hidden", 128)), alpha=h, seed=s).fit(fx["train"], Y["train"]) for s in seeds]
    if (~np.isnan(d_train)).sum() >= 2:
        fitted["external_difficulty"] = [ExternalDifficulty().fit(fx["train"], Y["train"], d_train)]
    else:
        log.warning("external_difficulty skipped: no RouterBench labels in train")
    log.info("hyper-parameters (val Brier): %s", chosen)

    use_platt = bool(r.get("platt", True))
    P: dict[str, list[dict[str, np.ndarray]]] = {}
    calibration = {}
    for name, ests in fitted.items():
        P[name] = []
        for i, est in enumerate(ests):
            raw_val = est.predict(fx["val"])
            platt = Platt().fit(raw_val, Y["val"]) if use_platt else None
            P[name].append({s: (platt.predict(est.predict(fx[s])) if platt else est.predict(fx[s])) for s in fx if s != "train"})
            if i == 0:
                calibration[name] = {v: {"raw_val": reliability(raw_val[:, j], Y["val"][:, j]),
                                         "platt_val": reliability(P[name][0]["val"][:, j], Y["val"][:, j])}
                                     for j, v in enumerate(variants)}
    run.write_json("calibration.json", clean(calibration))
    log.info("val Brier after scaling: %s", {n: round(float(((P[n][0]["val"] - Y["val"]) ** 2).mean()), 4) for n in P})

    # Router overhead: one query at a time on the CPU, as on an edge device.
    probe = texts[:1]
    probe_n = used["n_choices"].to_numpy()[:1]
    def full_features():
        return Features(encoder.encode(probe, device="cpu"), handcrafted(probe, probe_n), feats.task[:1])
    overhead = {"task_table": 0.0}
    overhead["rule_table"] = measure_overhead(lambda: fitted["rule_table"][0].predict(Features(emb[:1], handcrafted(probe, probe_n), feats.task[:1])))
    for name in ("logreg", "mlp", "knn", "external_difficulty"):
        if name in fitted:
            overhead[name] = measure_overhead(lambda: fitted[name][0].predict(full_features()))
    run.write_json("overhead_ms.json", overhead)
    log.info("router overhead (ms per query, CPU, batch 1): %s", {k: round(v, 2) for k, v in overhead.items()})

    # --- cost models -------------------------------------------------------------------------------------------
    names = cfg["variant_profile_names"]
    cost_models = {label: load_profiles(spec, names) for label, spec in cfg["profiles"].items()}
    if get_dotted(cfg, "param_ratio.base") and get_dotted(cfg, "param_ratio.base") in cost_models:
        cost_models["param_ratio"] = param_ratio_profiles(cost_models[cfg["param_ratio"]["base"]], cfg["outcomes"]["trainval"], variants)
    lo, hi, n = get_dotted(cfg, "lambda_grid", [-3, 1, 81])
    lambdas = [0.0] + [float(x) for x in np.logspace(lo, hi, int(n))]
    gen_mean = expected_gen_tokens(n_gen["train"], is_mcq["train"])
    run.write_json("expected_gen_tokens.json", dict(zip(variants, gen_mean.tolist())))

    selection, summary = {}, []
    for label, prof in cost_models.items():
        long_c = {}
        for s in fx:
            sel = long_of[s][long_of[s]["query_id"].isin(set(qid[s]))].copy()
            sel["cost_ms"] = query_costs(sel, prof)
            long_c[s] = sel
        C = {s: wide(long_c[s], "cost_ms").loc[qid[s], variants].to_numpy(dtype=float) for s in fx}
        _, unit = predicted_cost(n_ctx["train"], is_mcq["train"], gen_mean, variants, prof)
        Ct = {s: predicted_cost(n_ctx[s], is_mcq[s], gen_mean, variants, prof, unit)[0] for s in fx}
        ref_val = float(Y["val"][:, 0].mean())
        task = {s: fx[s].task for s in fx}

        curves, res = [], {"cost_model": label, "variants": variants, "routers": {}}
        res["static_val"] = {v: {"accuracy": a, "cost_ms": c} for c, a, v in static_points(Y["val"], C["val"], variants)}
        if evaluate_test:
            pts = static_points(Y["test"], C["test"], variants)
            hull = upper_hull(pts)
            res["static_test"] = {v: {"accuracy": a, "cost_ms": c, "macro_accuracy": float(pd.Series(Y["test"][:, j]).groupby(task["test"]).mean().mean())}
                                  for j, (c, a, v) in enumerate(pts)}
            res["hull_test"] = [{"variant": v, "cost_ms": c, "accuracy": a} for c, a, v in hull]
            res["oracle_test"] = {k: v for k, v in analyze(long_c["test"], order=variants).items()
                                  if k in ("oracle", "informative_oracle", "hull_accuracy_at_oracle_cost", "headroom_points")}
            rng = np.random.default_rng(cfg["seed"])
            rnd = replay(rng.integers(0, len(variants), size=len(qid["test"])), Y["test"], C["test"], task["test"])
            res["uniform_random_test"] = clean(rnd)

        for name in P:
            over = overhead[name]
            per_seed = []
            for i, ps in enumerate(P[name]):
                cv = curve(ps["val"], Ct["val"], lambdas, Y["val"], C["val"], task["val"], over)
                curves.append(cv.assign(router=name, seed=i, split="val"))
                picks = {t: select(cv, t, ref_val) for t in TARGETS}
                selection.setdefault(label, {}).setdefault(name, []).append(picks)
                seed_res = {"lambda": picks}
                if evaluate_test:
                    ct = curve(ps["test"], Ct["test"], lambdas, Y["test"], C["test"], task["test"], over)
                    curves.append(ct.assign(router=name, seed=i, split="test"))
                    seed_res["area_above_hull_points"] = (lambda a: None if a is None else 100 * a)(area_above_hull(ct, hull))
                    for t, lam in picks.items():
                        if lam is None:
                            seed_res[t] = None
                            continue
                        pick = route(ps["test"], Ct["test"], lam)
                        rep = replay(pick, Y["test"], C["test"], task["test"], over)
                        gain = rep["accuracy"] - hull_accuracy(hull, rep["cost_ms"])
                        y_router = Y["test"][np.arange(len(pick)), pick]
                        seed_res[t] = {**rep, "val": curve(ps["val"], Ct["val"], [lam], Y["val"], C["val"], task["val"], over).iloc[0][["accuracy", "cost_ms"]].to_dict(),
                                       "gain_over_hull_points": 100 * gain,
                                       "saving_vs_unpruned": 1 - rep["cost_ms"] / float(C["test"][:, 0].mean()),
                                       "accuracy_vs_unpruned_points": 100 * (rep["accuracy"] - float(Y["test"][:, 0].mean())),
                                       "mcnemar_vs_unpruned": mcnemar(y_router, Y["test"][:, 0])}
                        if i == 0:
                            seed_res[t]["bootstrap"] = bootstrap(pick, Y["test"], C["test"], over, variants, int(cfg.get("bootstrap", 10000)), cfg["seed"])
                            if name != "task_table" and "task_table" in P:
                                lam_t = selection[label]["task_table"][0][t] if "task_table" in selection[label] else None
                                if lam_t is not None:
                                    pick_t = route(P["task_table"][0]["test"], Ct["test"], lam_t)
                                    seed_res[t]["mcnemar_vs_task_table"] = mcnemar(y_router, Y["test"][np.arange(len(pick_t)), pick_t])
                per_seed.append(seed_res)
            res["routers"][name] = {"overhead_ms": over, "seeds": per_seed}
            if evaluate_test:
                for t in TARGETS:
                    ok = [s[t] for s in per_seed if s.get(t)]
                    if not ok:
                        summary.append({"cost_model": label, "router": name, "target": t, "reachable_on_val": False})
                        continue
                    summary.append({
                        "cost_model": label, "router": name, "target": t, "reachable_on_val": True, "n_seeds": len(ok),
                        "accuracy": float(np.mean([s["accuracy"] for s in ok])), "accuracy_std": float(np.std([s["accuracy"] for s in ok])),
                        "macro_accuracy": float(np.mean([s["macro_accuracy"] for s in ok])),
                        "cost_ms": float(np.mean([s["cost_ms"] for s in ok])),
                        "saving_vs_unpruned": float(np.mean([s["saving_vs_unpruned"] for s in ok])),
                        "accuracy_vs_unpruned_points": float(np.mean([s["accuracy_vs_unpruned_points"] for s in ok])),
                        "gain_over_hull_points": float(np.nanmean([s["gain_over_hull_points"] for s in ok]))
                        if not all(math.isnan(s["gain_over_hull_points"]) for s in ok) else float("nan")})
        pd.concat(curves, ignore_index=True).to_csv(run.path(f"curves_{label}.csv"), index=False)
        run.write_json(f"results_{label}.json", clean(res))
        log.info("[%s] routers replayed", label)

    run.write_json("selection.json", clean(selection))
    run.write_json("hyperparameters.json", clean(chosen))
    if summary:
        tab = pd.DataFrame(summary)
        tab.to_csv(run.path("summary.csv"), index=False)
        log.info("test split:\n%s", tab.round(4).to_string())
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
