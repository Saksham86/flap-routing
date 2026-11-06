"""Per-variant success estimators: p_hat[q, v] = Pr[variant v answers q correctly | q].

All follow fit(features, Y) on the train split and predict(features) -> (n_queries, n_variants).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Features

EPS = 1e-4


def _fit_binary(make, X: np.ndarray, y: np.ndarray):
    """A variant that is always (or never) right on the fit set has only one class: predict its rate."""
    return float(y[0]) if y.min() == y.max() else make().fit(X, y)


def _predict_binary(model, X: np.ndarray) -> np.ndarray:
    return np.full(len(X), model) if isinstance(model, float) else model.predict_proba(X)[:, 1]


class TaskTable:
    """Task-type baseline: per-task accuracy of each variant. The only estimator that reads the task id."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, f: Features, Y: np.ndarray) -> "TaskTable":
        self.prior_ = Y.mean(axis=0)
        self.table_ = {t: (Y[f.task == t].sum(axis=0) + self.alpha * self.prior_) / ((f.task == t).sum() + self.alpha)
                       for t in np.unique(f.task)}
        return self

    def predict(self, f: Features) -> np.ndarray:
        return np.vstack([self.table_.get(t, self.prior_) for t in f.task])


class RuleTable:
    """Rule-based baseline: per-bucket accuracy, buckets = (MCQ?, has digits?, length tercile). No task id."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _bucket(self, f: Features) -> list[tuple]:
        length = np.searchsorted(self.cuts_, f.hand[:, 1])
        return list(zip((f.hand[:, 4] > 0).tolist(), (f.hand[:, 2] > 0).tolist(), length.tolist()))

    def fit(self, f: Features, Y: np.ndarray) -> "RuleTable":
        self.cuts_ = np.quantile(f.hand[:, 1], [1 / 3, 2 / 3])
        self.prior_ = Y.mean(axis=0)
        b = np.array(self._bucket(f), dtype=int)
        self.table_ = {}
        for key in {tuple(k) for k in b.tolist()}:
            m = (b == np.array(key)).all(axis=1)
            self.table_[key] = (Y[m].sum(axis=0) + self.alpha * self.prior_) / (m.sum() + self.alpha)
        return self

    def predict(self, f: Features) -> np.ndarray:
        return np.vstack([self.table_.get(tuple(int(x) for x in k), self.prior_) for k in self._bucket(f)])


class LogReg:
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C, self.seed = C, seed

    def fit(self, f: Features, Y: np.ndarray) -> "LogReg":
        make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000,  # noqa: E731
                                                                          random_state=self.seed))
        self.models_ = [_fit_binary(make, f.dense, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, f: Features) -> np.ndarray:
        return np.column_stack([_predict_binary(m, f.dense) for m in self.models_])


class MLP:
    """One small network with one sigmoid output per variant, so variants share a representation."""

    def __init__(self, hidden: int = 128, alpha: float = 1e-3, seed: int = 0, max_iter: int = 200):
        self.hidden, self.alpha, self.seed, self.max_iter = hidden, alpha, seed, max_iter

    def fit(self, f: Features, Y: np.ndarray) -> "MLP":
        self.scaler_ = StandardScaler().fit(f.dense)
        self.model_ = MLPClassifier((self.hidden,), alpha=self.alpha, early_stopping=True, n_iter_no_change=10,
                                    max_iter=self.max_iter, random_state=self.seed)
        self.model_.fit(self.scaler_.transform(f.dense), Y)
        return self

    def predict(self, f: Features) -> np.ndarray:
        return self.model_.predict_proba(self.scaler_.transform(f.dense))


class KNN:
    """Cosine kNN on the (unit-norm) embeddings, smoothed towards the variant's overall accuracy."""

    def __init__(self, k: int = 20, alpha: float = 1.0):
        self.k, self.alpha = k, alpha

    def fit(self, f: Features, Y: np.ndarray) -> "KNN":
        self.X_, self.Y_, self.prior_ = f.embedding.astype(np.float32), Y.astype(np.float32), Y.mean(axis=0)
        return self

    def predict(self, f: Features) -> np.ndarray:
        k = min(self.k, len(self.X_))
        out = []
        for start in range(0, len(f), 512):
            sim = f.embedding[start:start + 512].astype(np.float32) @ self.X_.T
            nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            out.append((self.Y_[nn].sum(axis=1) + self.alpha * self.prior_) / (k + self.alpha))
        return np.vstack(out)


class ExternalDifficulty:
    """Difficulty measured on other LLMs (RouterBench): query -> d_hat, then d_hat -> p_hat per variant.
    `difficulty` is NaN for queries RouterBench does not cover; they are left out of the regression only."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, f: Features, Y: np.ndarray, difficulty: np.ndarray) -> "ExternalDifficulty":
        have = ~np.isnan(difficulty)
        if have.sum() < 2:
            raise ValueError("no RouterBench labels among the training queries")
        self.reg_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(f.dense[have], difficulty[have])
        d = self.reg_.predict(f.dense)[:, None]
        self.models_ = [_fit_binary(lambda: LogisticRegression(max_iter=2000), d, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, f: Features) -> np.ndarray:
        d = self.reg_.predict(f.dense)[:, None]
        return np.column_stack([_predict_binary(m, d) for m in self.models_])


class Platt:
    """Per-variant logistic recalibration of p_hat. Fitted on val: the pruned variants were FLAP-calibrated on
    train questions, so their accuracy on train may not carry over exactly."""

    def fit(self, P: np.ndarray, Y: np.ndarray) -> "Platt":
        self.models_ = [_fit_binary(lambda: LogisticRegression(C=1e3, max_iter=1000), self._logit(P[:, j]), Y[:, j])
                        for j in range(P.shape[1])]
        return self

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))[:, None]

    def predict(self, P: np.ndarray) -> np.ndarray:
        return np.column_stack([_predict_binary(m, self._logit(P[:, j])) for j, m in enumerate(self.models_)])


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Brier score, expected calibration error and the reliability table for one variant."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    which = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            table.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "mean_true": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece), "table": table}
