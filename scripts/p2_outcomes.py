"""P2 step 2 (GPU): run ONE variant over every pool query with lm-eval; write per-query outcomes.

For each task family (resumable: a finished family's parquet is skipped on rerun):
  * select the pool docs (all splits by default), as SORTED doc indices per lm-eval task.
    lm-eval 0.4.13 maps logged results back with indices[position], which is only correct
    for a sorted list, so sorting is enforced here.
  * one simple_evaluate call per family with a fixed fewshot seed. GSM8K's 5-shot exemplars
    depend on the seed AND the evaluated doc list, so every variant must evaluate the same
    list. Each row stores prompt_hash so this can be verified across variants.
  * parse + integrity-check samples (src/qacr/data/outcomes.py); compare against lm-eval's aggregates.

Outputs in results/p2/<run_id>/:
  outcomes_<task>.parquet, task_<task>.json (timing + aggregate check), outcomes.parquet, summary.json

Usage: python scripts/p2_outcomes.py --config configs/p2_outcomes.yaml --set variant.name=v0 ...
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.data.outcomes import aggregate_check, parse_task_samples, to_frame  # noqa: E402
from qacr.data.tasks import DEFAULT_SPECS  # noqa: E402
from qacr.runtime.variants import count_params, load_variant  # noqa: E402


def select_docs(pool: pd.DataFrame, task: str, splits: list[str], max_docs: int | None) -> pd.DataFrame:
    sel = pool[(pool["task"] == task) & (pool["split"].isin(splits))].sort_values("query_id")
    if max_docs is not None:
        sel = sel.head(max_docs)
    return sel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed = cfg["seed"]
    seed_everything(seed)
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p2.outcomes", run.path("run.log"))

    pool_path = REPO / get_dotted(cfg, "pool.path", "data/p2/query_pool.parquet")
    pool = pd.read_parquet(pool_path)
    splits = list(get_dotted(cfg, "pool.splits", ["train", "val", "test"]))
    max_docs = get_dotted(cfg, "pool.max_docs_per_task")
    tasks = list(cfg.get("tasks") or sorted(pool["task"].unique()))
    log.info("pool %s: %d rows; tasks=%s splits=%s max_docs=%s", pool_path, len(pool), tasks, splits, max_docs)

    device = get_dotted(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if get_dotted(cfg, "require_gpu", False) and device != "cuda":
        log.error("require_gpu=true but no CUDA device")
        close_logger(log)
        return 2
    vspec = cfg["variant"]
    model, tok = load_variant(vspec, device)
    run.write_json("variant.json", {"spec": vspec, "params": count_params(model), "device": device})
    log.info("loaded variant %s (%s) params=%s", vspec["name"], vspec["path"], count_params(model))

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=get_dotted(cfg, "lm_eval.batch_size", 16),
              max_length=get_dotted(cfg, "lm_eval.max_length", 2048))
    tm = TaskManager()

    all_rows = []
    for task in tasks:
        out_path = run.path(f"outcomes_{task}.parquet")
        if out_path.exists():
            log.info("[%s] already done, loading", task)
            all_rows.append(pd.read_parquet(out_path))  # written with the fixed schema
            continue
        spec = DEFAULT_SPECS[task]
        sel = select_docs(pool, task, splits, max_docs)
        if sel.empty:
            log.warning("[%s] no docs selected, skipping", task)
            continue
        # consistency with the pool's recorded settings
        for col, want in (("metric", spec.metric), ("num_fewshot", spec.num_fewshot)):
            got = set(sel[col].unique())
            if got != {want}:
                raise ValueError(f"[{task}] pool column {col}={got} disagrees with spec {want}")
        samples, rows_by_lm_task = {}, defaultdict(dict)
        for lm_task, grp in sel.groupby("lm_task"):
            idx = sorted(int(i) for i in grp["doc_index"])
            if len(idx) != len(set(idx)):
                raise ValueError(f"[{lm_task}] duplicate doc indices")
            samples[lm_task] = idx
            for r in grp.to_dict("records"):
                rows_by_lm_task[lm_task][int(r["doc_index"])] = r
        gen_kwargs = None
        if spec.kind == "generate_until":
            gen_kwargs = {"max_gen_toks": int(get_dotted(cfg, "lm_eval.max_gen_toks", 256))}
        log.info("[%s] evaluating %d docs over %d lm-eval task(s), %d-shot", task, len(sel), len(samples),
                 spec.num_fewshot)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        res = simple_evaluate(
            model=lm, tasks=sorted(samples), num_fewshot=spec.num_fewshot, samples=samples,
            log_samples=True, bootstrap_iters=0, task_manager=tm, gen_kwargs=gen_kwargs,
            random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rows, checks = [], []
        for lm_task in sorted(samples):
            parsed = parse_task_samples(res["samples"][lm_task], rows_by_lm_task[lm_task], vspec["name"], tok,
                                        spec.kind, spec.gen_filter)
            rows.extend(parsed)
            checks.append(aggregate_check(parsed, res["results"], lm_task, spec.metric, spec.gen_filter))
        df = to_frame(rows)
        # lm-eval left-truncates contexts longer than max_length (minus the generation budget):
        # that would silently change the prompt, so count and flag any query at risk.
        max_len = int(get_dotted(cfg, "lm_eval.max_length", 2048))
        budget = max_len - (int(get_dotted(cfg, "lm_eval.max_gen_toks", 256)) if spec.kind == "generate_until" else 0)
        n_trunc = int((df["n_ctx_tokens"] > budget).sum())
        if n_trunc:
            log.warning("[%s] %d queries have context > %d tokens and may have been truncated", task, n_trunc, budget)
        df.to_parquet(out_path, index=False)
        acc = float(df["correct"].mean())
        run.write_json(f"task_{task}.json", {
            "task": task, "n": len(df), "accuracy": acc, "elapsed_s": elapsed,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else None,
            "aggregate_checks": checks, "n_truncation_risk": n_trunc, "context_budget_tokens": budget,
            "max_ctx_tokens": int(df["n_ctx_tokens"].max()), "num_fewshot": spec.num_fewshot, "metric": spec.metric,
            "gen_filter": spec.gen_filter,
        })
        log.info("[%s] n=%d accuracy=%.4f (%.1fs) aggregate checks passed", task, len(df), acc, elapsed)
        all_rows.append(df)

    if not all_rows:
        log.error("nothing evaluated")
        close_logger(log)
        return 1
    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(run.path("outcomes.parquet"), index=False)
    summary = {"variant": vspec["name"], "n": len(full),
               "per_task": {t: {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
                            for t, g in full.groupby("task")},
               "macro_accuracy": float(full.groupby("task")["correct"].mean().mean())}
    run.write_json("summary.json", summary)
    log.info("done: %s", summary)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
