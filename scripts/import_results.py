"""Import a run directory downloaded from Colab (zip) into results/<phase>/<run_id>/.

Usage: python scripts/import_results.py path/to/<run_id>.zip [--force]

The phase and run id are read from the zip's meta.json, not from the file name. Refuses to
overwrite an existing run unless --force, and rejects zip entries that would escape the
target directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results-root", default=str(REPO / "results"))
    args = ap.parse_args(argv)

    with zipfile.ZipFile(args.zip_path) as zf:
        names = zf.namelist()
        if "meta.json" not in names:
            print("error: meta.json not found at the top level of the zip", file=sys.stderr)
            return 1
        meta = json.loads(zf.read("meta.json"))
        phase, run_id = str(meta["phase"]), str(meta["run_id"])
        target = Path(args.results_root) / phase / run_id
        if target.exists():
            if not args.force:
                print(f"error: {target} exists (use --force to replace)", file=sys.stderr)
                return 1
            shutil.rmtree(target)
        target.mkdir(parents=True)
        root = target.resolve()
        for name in names:
            dest = (target / name).resolve()
            if root != dest and root not in dest.parents:
                print(f"error: unsafe path in zip: {name}", file=sys.stderr)
                shutil.rmtree(target)
                return 1
        zf.extractall(target)
    print(f"imported -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
