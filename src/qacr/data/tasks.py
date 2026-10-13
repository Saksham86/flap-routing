"""Task specifications and per-task field extraction for the query pool.

The pool is built from lm-eval's own task objects (``ConfigurableTask``), so document order,
gold labels and prompts are exactly what lm-eval evaluates. For each doc we keep:

  question_text  the user-facing question only (no template, no few-shot), the router's input
  context        a supporting passage when the task has one (BoolQ), else None
  choices        answer options for multiple-choice tasks, else None
  gold           gold choice index (MC) or the final numeric answer string (GSM8K)
  lm_doc_hash    lm-eval's own doc hash; the outcome parser asserts logged samples match it
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

MC = "multiple_choice"
GEN = "generate_until"


@dataclass(frozen=True)
class TaskSpec:
    name: str                     # our task family name (also the query_id prefix)
    lm_eval_name: str             # lm-eval task or group name
    kind: str                     # MC or GEN
    metric: str                   # primary correctness metric for this task
    num_fewshot: int
    cap: int | None = None        # max docs kept in the pool (None = all)
    stratify: str | None = None   # doc field used for proportional capping and splitting
    gen_filter: str | None = None # GEN only: which lm-eval filter defines correctness
    extract: Callable[[dict], dict] = field(default=lambda d: {}, repr=False, compare=False)


def _arc(doc: dict) -> dict:
    return {"question_text": doc["question"], "context": None, "choices": list(doc["choices"]["text"])}


def _hellaswag(doc: dict) -> dict:
    # lm-eval's process_docs builds "query" = activity label + context; "choices" = endings
    return {"question_text": doc["query"], "context": None, "choices": list(doc["choices"])}


def _winogrande(doc: dict) -> dict:
    return {"question_text": doc["sentence"], "context": None, "choices": [doc["option1"], doc["option2"]]}


def _piqa(doc: dict) -> dict:
    return {"question_text": doc["goal"], "context": None, "choices": [doc["sol1"], doc["sol2"]]}


def _boolq(doc: dict) -> dict:
    return {"question_text": doc["question"], "context": doc["passage"], "choices": ["no", "yes"]}


def _gsm8k(doc: dict) -> dict:
    return {"question_text": doc["question"], "context": None, "choices": None}


def _mmlu(doc: dict) -> dict:
    return {"question_text": doc["question"], "context": None, "choices": list(doc["choices"])}


# Primary metric per task follows the LLM-Pruner / FLAP reporting convention
# (acc_norm for HellaSwag and ARC-Challenge, acc elsewhere). Both acc and acc_norm are stored.
DEFAULT_SPECS: dict[str, TaskSpec] = {s.name: s for s in [
    TaskSpec("arc_easy", "arc_easy", MC, "acc", 0, extract=_arc),
    TaskSpec("arc_challenge", "arc_challenge", MC, "acc_norm", 0, extract=_arc),
    TaskSpec("hellaswag", "hellaswag", MC, "acc_norm", 0, cap=4000, extract=_hellaswag),
    TaskSpec("winogrande", "winogrande", MC, "acc", 0, extract=_winogrande),
    TaskSpec("piqa", "piqa", MC, "acc", 0, extract=_piqa),
    TaskSpec("boolq", "boolq", MC, "acc", 0, extract=_boolq),
    TaskSpec("mmlu", "mmlu", MC, "acc", 0, cap=4000, stratify="subject", extract=_mmlu),
    TaskSpec("gsm8k", "gsm8k", GEN, "exact_match", 5, gen_filter="flexible-extract", extract=_gsm8k),
]}


def specs_from_config(cfg_tasks: dict[str, dict] | None) -> dict[str, TaskSpec]:
    """Start from DEFAULT_SPECS; a config may drop tasks or override cap/metric/num_fewshot/gen_filter."""
    if not cfg_tasks:
        return dict(DEFAULT_SPECS)
    out = {}
    allowed = {"cap", "metric", "num_fewshot", "gen_filter"}
    for name, overrides in cfg_tasks.items():
        if name not in DEFAULT_SPECS:
            raise KeyError(f"Unknown task {name!r}; known: {sorted(DEFAULT_SPECS)}")
        overrides = overrides or {}
        bad = set(overrides) - allowed
        if bad:
            raise KeyError(f"Task {name}: unsupported override keys {sorted(bad)}")
        base = DEFAULT_SPECS[name]
        kw = {k: getattr(base, k) for k in base.__dataclass_fields__}
        kw.update(overrides)
        out[name] = TaskSpec(**kw)
    return out


def lm_doc_hash(doc: dict) -> str:
    """Exactly lm-eval 0.4.13's per-sample ``doc_hash`` (evaluator.py)."""
    from lm_eval.utils import handle_non_serializable, hash_string
    return hash_string(json.dumps(doc, indent=2, default=handle_non_serializable, ensure_ascii=False))


def gsm8k_final_answer(answer_field: str) -> str:
    """GSM8K gold: the text after the last '#### ', commas removed."""
    if "####" not in answer_field:
        raise ValueError("GSM8K answer has no '####' marker")
    return answer_field.split("####")[-1].strip().replace(",", "")


def gold_of(task: Any, doc: dict, spec: TaskSpec):
    """Gold label exactly as lm-eval's process_results derives it."""
    if spec.kind == GEN:
        return gsm8k_final_answer(task.doc_to_target(doc))
    # Mirrors ConfigurableTask.process_results (multiple_choice branch) in lm-eval 0.4.13:
    # doc_to_target already converts digit strings to int; a remaining str is a choice text.
    choices = task.doc_to_choice(doc)
    gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
    if isinstance(gold, str):
        if gold not in choices:
            raise ValueError(f"{spec.name}: gold string {gold!r} not among choices")
        gold = choices.index(gold)
    if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(choices):
        raise ValueError(f"{spec.name}: bad gold {gold!r} for {len(choices)} choices")
    return gold


def iter_leaf_tasks(task_dict: dict):
    """Flatten lm-eval's (possibly nested group) task dict into (task_name, ConfigurableTask)."""
    for key, val in task_dict.items():
        if isinstance(val, dict):
            yield from iter_leaf_tasks(val)
        else:
            name = key if isinstance(key, str) else getattr(key, "group_name", str(key))
            yield name, val
