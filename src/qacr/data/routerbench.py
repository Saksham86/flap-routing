"""RouterBench (withmartian/routerbench) loading and matching to our query pool.

RouterBench ships pickled pandas DataFrames. Unpickling can execute arbitrary code, so we
never use pandas.read_pickle on it. ``safe_load_pickle`` uses an Unpickler whose find_class
allows ONLY the exact globals below. These were found by statically scanning the 0-shot file
(revision 784021482c3f320c6619ed4b3bb3b41a21424fcb); anything else aborts the load.
"""
from __future__ import annotations

import io
import pickle
import re
from pathlib import Path

ROUTERBENCH_REPO = "withmartian/routerbench"
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_0SHOT_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"

ALLOWED_GLOBALS = frozenset({
    ("builtins", "slice"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
})


class UnsafePickleError(pickle.UnpicklingError):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED_GLOBALS:
            raise UnsafePickleError(f"blocked global {module}.{name}")
        # numpy 2 moved numpy.core -> numpy._core; resolve the allowed name there if needed
        if module.startswith("numpy.core."):
            import importlib
            try:
                return getattr(importlib.import_module("numpy._core." + module.split(".", 2)[2]), name)
            except (ImportError, AttributeError):
                pass
        return super().find_class(module, name)


def safe_load_pickle(path: str | Path, expected_sha256: str | None = None):
    data = Path(path).read_bytes()
    if expected_sha256 is not None:
        import hashlib
        got = hashlib.sha256(data).hexdigest()
        if got != expected_sha256:
            raise UnsafePickleError(f"sha256 mismatch: {got} != {expected_sha256}")
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def download_0shot(dest_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(ROUTERBENCH_REPO, "routerbench_0shot.pkl", repo_type="dataset",
                                revision=ROUTERBENCH_REVISION, local_dir=str(dest_dir)))


_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


# ---- prompt parsing --------------------------------------------------------------------
_OPT = re.compile(r"^([A-H])\)\s?(.*)$")
RB_TASKS = {  # our task -> how to recognise its RouterBench eval_name
    "mmlu": lambda e: e.startswith("mmlu-"),
    "hellaswag": lambda e: e == "hellaswag",
    "winogrande": lambda e: e == "winogrande",
    "arc_challenge": lambda e: e == "arc-challenge",
    "gsm8k": lambda e: e == "grade-school-math",
}


def prompt_text(raw) -> str:
    """RouterBench stores the prompt as the str() of a one-element Python list."""
    import ast
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        val = ast.literal_eval(raw)  # literals only, no code execution
        if isinstance(val, list) and len(val) >= 1:
            return str(val[-1])
    return str(raw)


def parse_mc_prompt(text: str) -> tuple[str, list[str]]:
    """Return (question stem, options) from a RouterBench multiple-choice prompt."""
    text = text.replace("Please answer with the letter of the correct answer.\n\n", "", 1)
    lines = text.split("\n")
    stem_lines, options = [], []
    for ln in lines:
        m = _OPT.match(ln)
        if m and (options or ln.startswith("A)")):
            if m.group(1) != "ABCDEFGH"[len(options)]:
                break
            options.append(m.group(2))
        elif options:
            break  # first non-option line after the options ends the block ("Print only ...")
        else:
            stem_lines.append(ln)
    return "\n".join(stem_lines).strip(), options


def parse_gsm8k_question(text: str) -> str:
    """Return the target question.

    RouterBench's GSM8K prompt is a multi-element list whose LAST element is the bare question
    (``prompt_text`` returns that element). If a 'Question:' label is present (single-string
    prompts with inline exemplars), the target is the LAST such block.
    """
    i = text.rfind("Question:")
    if i < 0:
        if not text.strip():
            raise ValueError("empty GSM8K prompt")
        return text.strip()
    q = text[i + len("Question:"):]
    j = q.find("\nAnswer")
    return (q[:j] if j >= 0 else q).strip()


def _hellaswag_pre(text: str) -> str:
    # identical to lm_eval/tasks/hellaswag/utils.py::preprocess (0.4.13)
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def match_key(task: str, stem: str, options: list[str] | None) -> str:
    """Content key for matching. Options are SORTED: RouterBench shuffles answer order
    relative to the source datasets (verified on Winogrande/MMLU/ARC). Its per-model scores
    are correctness flags, so they do not depend on option order."""
    if task == "gsm8k":
        return norm_text(stem)
    if task == "hellaswag":   # lm-eval rewrites the context, so key on preprocessed endings only
        return "\x1e".join(sorted(norm_text(_hellaswag_pre(o)) for o in options))
    return norm_text(stem) + "\x1f" + "\x1e".join(sorted(norm_text(o) for o in options))


def model_columns(rb) -> list[str]:
    return [c for c in rb.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name",
                                                                "oracle_model_to_route_to")]


def match_pool(pool, rb):
    """Attach RouterBench difficulty to pool rows by content. Returns (labels_df, report).

    labels_df columns: query_id, rb_sample_id, rb_mean_score, rb_n_models
    Keys that occur more than once in RouterBench with different scores are 'ambiguous'
    and left unmatched rather than guessed.
    """
    import numpy as np
    import pandas as pd

    models = model_columns(rb)
    report, out = {"models": models}, []
    for task, is_task in RB_TASKS.items():
        sub = rb[rb["eval_name"].map(is_task)]
        index: dict[str, list[tuple[str, float]]] = {}
        parse_fail = 0
        for r in sub.itertuples(index=False):
            text = prompt_text(r.prompt)
            try:
                if task == "gsm8k":
                    key = match_key(task, parse_gsm8k_question(text), None)
                else:
                    stem, opts = parse_mc_prompt(text)
                    if len(opts) < 2:
                        raise ValueError("no options")
                    key = match_key(task, stem, opts)
            except ValueError:
                parse_fail += 1
                continue
            row = dict(zip(sub.columns, r))
            vals = np.array([row[m] for m in models], dtype=float)
            index.setdefault(key, []).append((row["sample_id"], float(np.nanmean(vals)), int(np.isfinite(vals).sum())))
        prow = pool[pool["task"] == task]
        matched = ambiguous = 0
        for p in prow.itertuples(index=False):
            opts = list(p.choices) if p.choices is not None else None
            key = match_key(task, p.question_text, opts)
            hits = index.get(key)
            if not hits:
                continue
            if len({round(h[1], 9) for h in hits}) > 1:
                ambiguous += 1
                continue
            matched += 1
            out.append({"query_id": p.query_id, "rb_sample_id": hits[0][0], "rb_mean_score": hits[0][1],
                        "rb_n_models": hits[0][2]})
        report[task] = {"routerbench_rows": int(len(sub)), "parse_failures": parse_fail,
                        "pool_rows": int(len(prow)), "matched": matched, "ambiguous": ambiguous,
                        "match_rate": (matched / len(prow)) if len(prow) else None}
    for task in sorted(set(pool["task"]) - set(RB_TASKS)):
        report[task] = {"routerbench_rows": 0, "pool_rows": int((pool["task"] == task).sum()), "matched": 0,
                        "note": "task not in RouterBench"}
    return pd.DataFrame(out, columns=["query_id", "rb_sample_id", "rb_mean_score", "rb_n_models"]), report
