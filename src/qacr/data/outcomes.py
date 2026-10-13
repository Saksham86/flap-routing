"""Turn lm-eval logged samples into per-(query, variant) outcome rows, with integrity checks.

Checks (any failure raises; nothing is silently dropped):
  * every logged doc maps to a pool row (via doc_id) AND its lm-eval doc_hash equals the pool's
  * every selected pool doc appears exactly once (per filter for generation tasks)
  * MC: our recomputed acc / acc_norm from the logged log-likelihoods equals lm-eval's per-sample value
  * the per-task mean of the primary metric equals lm-eval's aggregate (checked by the caller)

Columns written per row:
  query_id, variant, task, lm_task, correct (primary metric, 0/1), acc, acc_norm,
  soft (prob. of gold under the primary normalisation), pred, gold, lls (MC log-likelihoods),
  exact_match_strict / exact_match_flex (GSM8K), gen_text, gen_extracted, n_ctx_tokens,
  n_gen_tokens, prompt_hash
"""
from __future__ import annotations

from typing import Any

import numpy as np


class OutcomeIntegrityError(AssertionError):
    pass


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def mc_scores(lls: list[float], lm_choices: list[str], gold: int) -> dict:
    """Recompute lm-eval's acc / acc_norm (and our soft scores) from choice log-likelihoods."""
    lls_a = np.asarray(lls, dtype=np.float64)
    if len(lls_a) != len(lm_choices):
        raise OutcomeIntegrityError(f"{len(lls_a)} log-likelihoods for {len(lm_choices)} choices")
    lens = np.asarray([float(len(c)) for c in lm_choices])
    pred = int(np.argmax(lls_a))
    pred_norm = int(np.argmax(lls_a / lens))
    return {
        "pred": pred, "pred_norm": pred_norm,
        "acc": float(pred == gold), "acc_norm": float(pred_norm == gold),
        "soft_acc": float(_softmax(lls_a)[gold]),
        "soft_acc_norm": float(_softmax(lls_a / lens)[gold]),
    }


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def parse_task_samples(samples: list[dict], pool_rows: dict[int, dict], variant: str, tokenizer,
                       kind: str, gen_filter: str | None = None) -> list[dict]:
    """``pool_rows``: doc_index -> pool row (only the docs that were evaluated)."""
    by_doc: dict[int, dict[str, dict]] = {}
    for s in samples:
        doc_id = int(s["doc_id"])
        if doc_id not in pool_rows:
            raise OutcomeIntegrityError(f"logged doc_id {doc_id} was not selected")
        row = pool_rows[doc_id]
        if s["doc_hash"] != row["lm_doc_hash"]:
            raise OutcomeIntegrityError(
                f"{row['query_id']}: logged doc_hash != pool doc_hash (doc_id mapping is wrong?)")
        filt = s.get("filter", "none")
        if filt in by_doc.setdefault(doc_id, {}):
            raise OutcomeIntegrityError(f"{row['query_id']}: duplicate sample for filter {filt}")
        by_doc[doc_id][filt] = s
    missing = sorted(set(pool_rows) - set(by_doc))
    if missing:
        raise OutcomeIntegrityError(f"{len(missing)} selected docs have no logged sample, e.g. {missing[:5]}")

    out = []
    for doc_id in sorted(by_doc):
        row, per_filter = pool_rows[doc_id], by_doc[doc_id]
        base = {"query_id": row["query_id"], "variant": variant, "task": row["task"], "lm_task": row["lm_task"]}
        if kind == "multiple_choice":
            s = per_filter.get("none") or next(iter(per_filter.values()))
            lls = [float(r[0]) for r in s["filtered_resps"]]
            gold = int(row["gold"])
            sc = mc_scores(lls, list(row["lm_choices"]), gold)
            for m in ("acc", "acc_norm"):
                if m in s and float(s[m]) != sc[m]:
                    raise OutcomeIntegrityError(f"{row['query_id']}: recomputed {m}={sc[m]} != lm-eval {s[m]}")
            metric = row["metric"]
            if metric not in ("acc", "acc_norm"):
                raise OutcomeIntegrityError(f"unsupported MC metric {metric}")
            ctx = s["arguments"][0][0]
            out.append({**base, "correct": int(sc[metric]), "acc": sc["acc"], "acc_norm": sc["acc_norm"],
                        "soft": sc[f"soft_{metric}"], "pred": sc["pred_norm" if metric == "acc_norm" else "pred"],
                        "gold": gold, "lls": lls,
                        "exact_match_strict": None, "exact_match_flex": None, "gen_text": None,
                        "gen_extracted": None, "n_ctx_tokens": _n_tokens(tokenizer, ctx), "n_gen_tokens": 0,
                        "prompt_hash": s["prompt_hash"]})
        elif kind == "generate_until":
            if gen_filter not in per_filter:
                raise OutcomeIntegrityError(f"{row['query_id']}: filter {gen_filter!r} missing; have {list(per_filter)}")
            primary = per_filter[gen_filter]
            gen_text = primary["resps"][0][0]
            em = {f: float(per_filter[f]["exact_match"]) if f in per_filter else None
                  for f in ("strict-match", "flexible-extract")}
            ctx = primary["arguments"][0][0]
            out.append({**base, "correct": int(float(primary["exact_match"])), "acc": None, "acc_norm": None,
                        "soft": None, "pred": None, "gold": row["gold"], "lls": None,
                        "exact_match_strict": em["strict-match"], "exact_match_flex": em["flexible-extract"],
                        "gen_text": gen_text, "gen_extracted": str(primary["filtered_resps"][0]),
                        "n_ctx_tokens": _n_tokens(tokenizer, ctx), "n_gen_tokens": _n_tokens(tokenizer, gen_text),
                        "prompt_hash": primary["prompt_hash"]})
        else:
            raise ValueError(f"unknown task kind {kind}")
    return out


def aggregate_check(rows: list[dict], lm_results: dict[str, Any], lm_task: str, metric: str,
                    gen_filter: str | None, atol: float = 1e-9) -> dict:
    """Compare our mean of the primary metric with lm-eval's aggregate for one lm task."""
    key = f"{metric},{gen_filter or 'none'}"
    if lm_task not in lm_results or key not in lm_results[lm_task]:
        raise OutcomeIntegrityError(f"lm-eval result {lm_task}/{key} not found")
    ours = float(np.mean([r["correct"] for r in rows if r["lm_task"] == lm_task]))
    theirs = float(lm_results[lm_task][key])
    if abs(ours - theirs) > atol:
        raise OutcomeIntegrityError(f"{lm_task}: our mean {ours} != lm-eval {key} {theirs}")
    return {"lm_task": lm_task, "metric": key, "ours": ours, "lm_eval": theirs}


OUTCOME_SCHEMA = {
    "query_id": "string", "variant": "string", "task": "string", "lm_task": "string",
    "correct": "int8", "acc": "Float64", "acc_norm": "Float64", "soft": "Float64",
    "pred": "Int64", "gold": "string", "exact_match_strict": "Float64", "exact_match_flex": "Float64",
    "gen_text": "string", "gen_extracted": "string", "n_ctx_tokens": "int32", "n_gen_tokens": "int32",
    "prompt_hash": "string",
}  # "lls" stays an object column: list[float] for MC, None for generation


def to_frame(rows: list[dict]):
    """Rows -> DataFrame with one fixed schema, so per-task files concatenate cleanly."""
    import pandas as pd
    df = pd.DataFrame(rows)
    missing = set(OUTCOME_SCHEMA) | {"lls"}
    missing -= set(df.columns)
    if missing:
        raise OutcomeIntegrityError(f"outcome rows missing columns {sorted(missing)}")
    df["gold"] = df["gold"].map(lambda g: None if g is None or (isinstance(g, float) and g != g) else str(g))
    df = df.astype(OUTCOME_SCHEMA)
    df["lls"] = df["lls"].map(lambda v: None if v is None else [float(x) for x in v])
    return df[list(OUTCOME_SCHEMA) + ["lls"]]
