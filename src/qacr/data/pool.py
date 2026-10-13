"""Query pool construction: dedupe -> cap (proportional by stratum) -> seeded 60/20/20 splits.

Pure functions (``dedupe_key``, ``proportional_quota``, ``assign_splits``) are unit tested.
``build_pool`` instantiates lm-eval tasks and needs network access the first time.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .tasks import GEN, TaskSpec, gold_of, iter_leaf_tasks, lm_doc_hash

SPLITS = ("train", "val", "test")
_WS = re.compile(r"\s+")


def _norm(s: str | None) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def dedupe_key(question_text: str, context: str | None, choices: list[str] | None) -> str:
    return "\x1f".join([_norm(question_text), _norm(context), "\x1e".join(_norm(c) for c in (choices or []))])


def proportional_quota(stratum_sizes: dict[str, int], cap: int) -> dict[str, int]:
    """Largest-remainder allocation of ``cap`` items proportional to stratum sizes.

    Deterministic: ties in the remainder are broken by stratum name. Never allocates more
    than a stratum has. If cap >= total, every stratum keeps everything.
    """
    total = sum(stratum_sizes.values())
    if cap >= total:
        return dict(stratum_sizes)
    if cap < 0:
        raise ValueError("cap must be >= 0")
    exact = {k: cap * n / total for k, n in stratum_sizes.items()}
    quota = {k: min(int(np.floor(v)), stratum_sizes[k]) for k, v in exact.items()}
    remaining = cap - sum(quota.values())
    order = sorted(stratum_sizes, key=lambda k: (-(exact[k] - np.floor(exact[k])), k))
    i = 0
    while remaining > 0:
        k = order[i % len(order)]
        if quota[k] < stratum_sizes[k]:
            quota[k] += 1
            remaining -= 1
        i += 1
    return quota


def split_sizes(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) < 0:
        raise ValueError(f"split fractions must be >= 0 and sum to 1, got {fractions}")
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    n_val = min(n_val, n - n_train)
    return n_train, n_val, n - n_train - n_val


def assign_splits(ids_by_stratum: dict[str, list[str]], fractions: tuple[float, float, float],
                  rng: np.random.Generator) -> dict[str, str]:
    """Shuffle each stratum (in sorted stratum order, ids sorted first) and cut 60/20/20."""
    out: dict[str, str] = {}
    for stratum in sorted(ids_by_stratum):
        ids = sorted(ids_by_stratum[stratum])
        perm = rng.permutation(len(ids))
        n_train, n_val, _ = split_sizes(len(ids), fractions)
        for rank, j in enumerate(perm):
            out[ids[j]] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return out


@dataclass
class PoolResult:
    rows: list[dict]
    report: dict


def _query_id(task: str, lm_task: str, idx: int) -> str:
    return f"{task}/{lm_task}/{idx:05d}"


def build_pool(specs: dict[str, TaskSpec], seed: int, fractions=(0.6, 0.2, 0.2), task_manager=None,
               log=print) -> PoolResult:
    from lm_eval.tasks import TaskManager, get_task_dict

    tm = task_manager or TaskManager()
    rows: list[dict] = []
    report: dict = {"tasks": {}}
    for name in sorted(specs):
        spec = specs[name]
        rng = np.random.default_rng([seed, _stable_int(name)])
        leaves = list(iter_leaf_tasks(get_task_dict([spec.lm_eval_name], tm)))
        task_rows: list[dict] = []
        n_eval, n_dupe = 0, 0
        seen: set[str] = set()  # dedupe across all leaf tasks of a family (e.g. all MMLU subjects)
        for lm_task, task in sorted(leaves, key=lambda x: x[0]):
            if task.OUTPUT_TYPE != spec.kind:
                raise ValueError(f"{lm_task}: output type {task.OUTPUT_TYPE} != spec {spec.kind}")
            for idx, doc in enumerate(task.eval_docs):
                n_eval += 1
                fields = spec.extract(doc)
                key = dedupe_key(fields["question_text"], fields["context"], fields["choices"])
                if key in seen:
                    n_dupe += 1
                    continue
                seen.add(key)
                gold = gold_of(task, doc, spec)
                stratum = str(doc[spec.stratify]) if spec.stratify else name
                task_rows.append({
                    "query_id": _query_id(name, lm_task, idx),
                    "task": name, "lm_task": lm_task, "doc_index": idx, "stratum": stratum,
                    "question_text": fields["question_text"], "context": fields["context"],
                    "choices": fields["choices"],
                    "n_choices": len(fields["choices"]) if fields["choices"] is not None else 0,
                    # lm-eval's own choice strings; acc_norm divides log-likelihoods by their
                    # character length (for Winogrande these are the two contexts, not options)
                    "lm_choices": list(task.doc_to_choice(doc)) if spec.kind != GEN else None,
                    "gold": str(gold), "metric": spec.metric, "gen_filter": spec.gen_filter,
                    "num_fewshot": spec.num_fewshot, "lm_doc_hash": lm_doc_hash(doc),
                })
        by_stratum: dict[str, list[dict]] = defaultdict(list)
        for r in task_rows:
            by_stratum[r["stratum"]].append(r)
        if spec.cap is not None:
            quota = proportional_quota({k: len(v) for k, v in by_stratum.items()}, spec.cap)
            kept = []
            for stratum in sorted(by_stratum):
                group = sorted(by_stratum[stratum], key=lambda r: r["query_id"])
                pick = rng.choice(len(group), size=quota[stratum], replace=False) if quota[stratum] < len(group) \
                    else np.arange(len(group))
                kept.extend(group[i] for i in sorted(pick))
            task_rows = kept
        ids_by_stratum: dict[str, list[str]] = defaultdict(list)
        for r in task_rows:
            ids_by_stratum[r["stratum"]].append(r["query_id"])
        split_of = assign_splits(ids_by_stratum, fractions, rng)
        for r in task_rows:
            r["split"] = split_of[r["query_id"]]
        rows.extend(task_rows)
        counts = Counter(r["split"] for r in task_rows)
        report["tasks"][name] = {
            "lm_eval_name": spec.lm_eval_name, "n_leaf_tasks": len(leaves), "n_eval_docs": n_eval,
            "n_duplicates_dropped": n_dupe, "n_kept": len(task_rows), "cap": spec.cap,
            "splits": {s: counts.get(s, 0) for s in SPLITS}, "metric": spec.metric,
            "num_fewshot": spec.num_fewshot, "gen_filter": spec.gen_filter,
            "dataset": _dataset_ids(leaves),
        }
        log(f"{name}: eval_docs={n_eval} dupes={n_dupe} kept={len(task_rows)} splits={dict(counts)}")
    rows.sort(key=lambda r: r["query_id"])
    ids = [r["query_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate query_id")
    report["n_total"] = len(rows)
    report["splits_total"] = dict(Counter(r["split"] for r in rows))
    return PoolResult(rows=rows, report=report)


def _stable_int(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _dataset_ids(leaves) -> dict:
    """Record dataset path/name and the Hub commit sha it resolved to (for reproducibility)."""
    first = leaves[0][1]
    path, name = first.config.dataset_path, first.config.dataset_name
    import time

    from huggingface_hub import HfApi
    sha, err = None, None
    for attempt in range(6):  # the Hub occasionally resets connections; retry with backoff
        try:
            sha = HfApi().dataset_info(path).sha
            break
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    if sha is None:
        sha = f"unavailable: {err.__class__.__name__}"
    return {"path": path, "name": name if len(leaves) == 1 else f"{len(leaves)} configs", "hub_sha": sha}
