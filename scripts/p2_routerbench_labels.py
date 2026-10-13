"""P2 step 1b (CPU): attach RouterBench 'external difficulty' to pool queries (P7 ablation input).

Downloads routerbench_0shot.pkl at a pinned revision into an isolated directory, verifies its
SHA-256, loads it with the restricted unpickler (never pandas.read_pickle), and matches rows to
our pool by content. Unmatched/ambiguous queries get no label. Match rates are reported per task.

Outputs in results/p2/<run_id>/: routerbench_labels.parquet, routerbench_report.json
--publish copies them next to the published pool (paths.publish_dir).
"""
from __future__ import annotations

import argparse
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
from qacr.data.routerbench import (ROUTERBENCH_0SHOT_SHA256, ROUTERBENCH_REVISION, download_0shot,  # noqa: E402
                                   match_pool, safe_load_pickle)


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
    log = get_logger("p2.routerbench", run.path("run.log"))

    publish_dir = REPO / get_dotted(cfg, "paths.publish_dir", "data/p2")
    pool = pd.read_parquet(publish_dir / "query_pool.parquet")
    dl_dir = Path(get_dotted(cfg, "routerbench.download_dir") or (REPO / ".cache" / "routerbench"))
    dl_dir.mkdir(parents=True, exist_ok=True)
    pkl = download_0shot(dl_dir)
    rb = safe_load_pickle(pkl, ROUTERBENCH_0SHOT_SHA256)
    log.info("RouterBench 0-shot: %s rows (revision %s)", rb.shape, ROUTERBENCH_REVISION)

    labels, report = match_pool(pool, rb)
    report["revision"] = ROUTERBENCH_REVISION
    report["sha256"] = ROUTERBENCH_0SHOT_SHA256
    report["n_labelled"] = int(len(labels))
    labels.to_parquet(run.path("routerbench_labels.parquet"), index=False)
    run.write_json("routerbench_report.json", report)
    for task, r in report.items():
        if isinstance(r, dict) and "pool_rows" in r:
            log.info("%-14s pool=%5d matched=%5d ambiguous=%s rb_rows=%s", task, r["pool_rows"], r["matched"],
                     r.get("ambiguous", "-"), r["routerbench_rows"])

    if args.publish:
        for name in ("routerbench_labels.parquet", "routerbench_report.json"):
            dest = publish_dir / name
            if dest.exists() and not args.force:
                log.error("%s exists; use --force", dest)
                close_logger(log)
                return 1
            shutil.copy2(run.path(name), dest)
        log.info("published -> %s", publish_dir)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
