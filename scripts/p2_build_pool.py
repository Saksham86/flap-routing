"""P2 step 1 (CPU): build the query pool + splits from lm-eval's task definitions.

Writes to results/p2/<run_id>/:
  query_pool.parquet            one row per query (see src/qacr/data/tasks.py for columns)
  splits/{train,val,test}.txt   query_ids, one per line, sorted
  manifest.json                 per-task counts, dataset Hub shas, versions, file hashes
With --publish, also copies these to paths.publish_dir (default data/p2/), refusing to
overwrite an existing publication unless --force.

Usage: python scripts/p2_build_pool.py --config configs/p2_pool.yaml [--publish [--force]]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.data.pool import SPLITS, build_pool  # noqa: E402
from qacr.data.tasks import specs_from_config  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p2.pool", run.path("run.log"))

    specs = specs_from_config(cfg.get("tasks"))
    fractions = tuple(cfg.get("split_fractions", (0.6, 0.2, 0.2)))
    res = build_pool(specs, seed=cfg["seed"], fractions=fractions, log=log.info)

    df = pd.DataFrame(res.rows)
    pool_path = run.path("query_pool.parquet")
    df.to_parquet(pool_path, index=False)
    split_dir = run.path("splits")
    split_dir.mkdir(exist_ok=True)
    for s in SPLITS:
        ids = sorted(df.loc[df["split"] == s, "query_id"])
        (split_dir / f"{s}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    import lm_eval
    manifest = {
        **res.report,
        "seed": cfg["seed"], "split_fractions": list(fractions),
        "lm_eval_version": lm_eval.__version__,
        "files": {"query_pool.parquet": sha256_file(pool_path),
                  **{f"splits/{s}.txt": sha256_file(split_dir / f"{s}.txt") for s in SPLITS}},
        "run_id": run.run_id,
    }
    run.write_json("manifest.json", manifest)
    log.info("pool: %d queries, splits %s", manifest["n_total"], manifest["splits_total"])

    if args.publish:
        missing_sha = [t for t, r in manifest["tasks"].items() if str(r["dataset"]["hub_sha"]).startswith("unavailable")]
        if missing_sha:
            log.error("refusing to publish: dataset Hub sha missing for %s (rerun when online)", missing_sha)
            close_logger(log)
            return 1
        dest = REPO / get_dotted(cfg, "paths.publish_dir", "data/p2")
        if dest.exists() and any(dest.iterdir()):
            if not args.force:
                log.error("%s already has a published pool; use --force to replace it", dest)
                close_logger(log)
                return 1
            shutil.rmtree(dest)
        (dest / "splits").mkdir(parents=True, exist_ok=True)
        shutil.copy2(pool_path, dest / "query_pool.parquet")
        shutil.copy2(run.path("manifest.json"), dest / "manifest.json")
        for s in SPLITS:
            shutil.copy2(split_dir / f"{s}.txt", dest / "splits" / f"{s}.txt")
        log.info("published -> %s", dest)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
