"""Query features for routers. Only the query's own text: never few-shot exemplars or templates, and
never the task id (only the task-type baseline sees that, explicitly)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_OPERATORS = re.compile(r"[+\-*/=%^<>]")


def query_text(pool: pd.DataFrame) -> list[str]:
    """context + question + answer options, in that order."""
    out = []
    for ctx, q, opts in zip(pool["context"], pool["question_text"], pool["choices"]):
        parts = [] if ctx is None or (isinstance(ctx, float) and np.isnan(ctx)) else [str(ctx)]
        parts.append("" if q is None else str(q))
        if opts is not None and len(opts):
            parts.extend(str(o) for o in opts)
        out.append(" ".join(parts))
    return out


def handcrafted(texts: list[str], n_choices: np.ndarray) -> np.ndarray:
    """log chars, log words, log digits, log operators, MCQ flag, option count."""
    rows = [(np.log1p(len(t)), np.log1p(len(t.split())), np.log1p(sum(c.isdigit() for c in t)),
             np.log1p(len(_OPERATORS.findall(t)))) for t in texts]
    n_choices = np.asarray(n_choices, dtype=float)
    return np.column_stack([np.asarray(rows, dtype=float).reshape(len(texts), 4), n_choices > 0, n_choices])


class MiniLMEncoder:
    """Sentence embeddings. The resolved Hub commit is exposed so that each run can record it."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", revision: str | None = None,
                 device: str | None = None):
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        path = snapshot_download(name, revision=revision)
        self.name, self.commit, self.device = name, path.replace("\\", "/").rsplit("/", 1)[-1], device
        self.model = SentenceTransformer(path, device=device)

    def encode(self, texts: list[str], device: str | None = None) -> np.ndarray:
        return self.model.encode(texts, batch_size=256 if len(texts) > 1 else 1, device=device or self.device,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


class HashingEncoder:
    """Deterministic offline stand-in for tests; not a thesis router feature."""

    def __init__(self, n_features: int = 64):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.name, self.commit = "hashing", None
        self.vec = HashingVectorizer(n_features=n_features, norm="l2", alternate_sign=False)

    def encode(self, texts: list[str], device: str | None = None) -> np.ndarray:
        return self.vec.transform(texts).toarray()


def make_encoder(spec: dict):
    kind = spec.get("kind", "minilm")
    if kind == "minilm":
        return MiniLMEncoder(spec.get("name", "sentence-transformers/all-MiniLM-L6-v2"), spec.get("revision"),
                             spec.get("device"))
    if kind == "hashing":
        return HashingEncoder(int(spec.get("n_features", 64)))
    raise ValueError(f"unknown encoder kind {kind!r}")


@dataclass
class Features:
    embedding: np.ndarray
    hand: np.ndarray
    task: np.ndarray

    def __len__(self) -> int:
        return len(self.task)

    def take(self, idx: np.ndarray) -> "Features":
        return Features(self.embedding[idx], self.hand[idx], self.task[idx])

    @property
    def dense(self) -> np.ndarray:
        return np.hstack([self.embedding, self.hand])
