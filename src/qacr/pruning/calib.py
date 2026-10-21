"""In-domain calibration text for FLAP: questions from the query pool's TRAIN split (no answers).

Motivation (measured in P2): FLAP calibrated on WikiText-2 produced variants that lose the question->answer
format (constant answers on MMLU/BoolQ, unfinished GSM8K answers). FLAP's importance scores come from the
activations on the calibration text, so calibrating on text in the evaluation format should keep the
channels used for it.

Leakage rules: only rows with split == "train" are used; only the question, its context and the answer
OPTIONS are included, never the gold answer. Sampling then follows FLAP's procedure exactly: concatenate,
tokenize, take `nsamples` random windows of `seqlen` tokens with random.seed(seed).
"""
from __future__ import annotations

import random

import pandas as pd


def format_question(row: dict) -> str:
    parts = []
    if row.get("context"):
        parts.append(str(row["context"]).strip())
    parts.append(f"Question: {str(row['question_text']).strip()}")
    choices = row.get("choices")
    if choices is not None and len(choices) > 0:
        for i, c in enumerate(choices):
            parts.append(f"{'ABCDEFGH'[i]}. {str(c).strip()}")
    parts.append("Answer:")
    return "\n".join(parts)


def pool_calibration(tokenizer, pool: pd.DataFrame, nsamples: int, seed: int, seqlen: int,
                     splits: tuple[str, ...] = ("train",)):
    rows = pool[pool["split"].isin(splits)].sort_values("query_id")
    if rows.empty:
        raise ValueError(f"no pool rows in splits {splits}")
    texts = [format_question(r) for r in rows.to_dict("records")]
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)                     # mix tasks, deterministically
    enc = tokenizer("\n\n".join(texts[i] for i in order), return_tensors="pt").input_ids
    random.seed(seed)                                      # FLAP's window sampling (lib/data.py)
    calib = []
    for _ in range(nsamples):
        i = random.randint(0, enc.shape[1] - seqlen - 1)
        calib.append(enc[:, i:i + seqlen])
    return calib, {"n_questions": len(texts), "n_tokens": int(enc.shape[1]), "splits": list(splits)}
