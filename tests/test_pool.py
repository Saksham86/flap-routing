from collections import Counter

import numpy as np
import pytest

from qacr.data.pool import assign_splits, dedupe_key, proportional_quota, split_sizes
from qacr.data.tasks import DEFAULT_SPECS, gsm8k_final_answer, specs_from_config


def test_quota_sums_and_bounds():
    sizes = {"a": 100, "b": 50, "c": 7, "d": 1}
    q = proportional_quota(sizes, 60)
    assert sum(q.values()) == 60
    assert all(0 <= q[k] <= sizes[k] for k in sizes)
    assert q == proportional_quota(sizes, 60)                 # deterministic
    assert proportional_quota(sizes, 1000) == sizes          # cap above total keeps all


def test_quota_proportional():
    q = proportional_quota({"x": 300, "y": 100}, 100)
    assert q == {"x": 75, "y": 25}


def test_quota_never_exceeds_small_stratum():
    q = proportional_quota({"big": 1000, "tiny": 1}, 999)
    assert q["tiny"] <= 1 and sum(q.values()) == 999


def test_split_sizes():
    assert split_sizes(10, (0.6, 0.2, 0.2)) == (6, 2, 2)
    assert sum(split_sizes(7, (0.6, 0.2, 0.2))) == 7
    assert split_sizes(1, (0.6, 0.2, 0.2))[0] == 1
    with pytest.raises(ValueError):
        split_sizes(10, (0.5, 0.2, 0.2))


def test_assign_splits_deterministic_and_disjoint():
    ids = {"s1": [f"a{i}" for i in range(50)], "s2": [f"b{i}" for i in range(30)]}
    a = assign_splits(ids, (0.6, 0.2, 0.2), np.random.default_rng(1))
    b = assign_splits(ids, (0.6, 0.2, 0.2), np.random.default_rng(1))
    assert a == b
    assert set(a) == set(ids["s1"]) | set(ids["s2"])
    c = Counter(v for k, v in a.items() if k.startswith("a"))
    assert c == {"train": 30, "val": 10, "test": 10}
    # input order must not matter (ids are sorted inside)
    rev = {k: list(reversed(v)) for k, v in ids.items()}
    assert assign_splits(rev, (0.6, 0.2, 0.2), np.random.default_rng(1)) == a


def test_dedupe_key_normalises_whitespace_and_case():
    assert dedupe_key("What  is X?", None, ["A", "b"]) == dedupe_key("what is x? ", "", ["a", " B"])
    assert dedupe_key("q", None, ["a", "b"]) != dedupe_key("q", None, ["b", "a"])


def test_gsm8k_answer():
    assert gsm8k_final_answer("blah\n#### 1,234") == "1234"
    with pytest.raises(ValueError):
        gsm8k_final_answer("no marker")


def test_specs_overrides():
    s = specs_from_config({"mmlu": {"cap": 10}, "gsm8k": {}})
    assert set(s) == {"mmlu", "gsm8k"} and s["mmlu"].cap == 10 and s["mmlu"].stratify == "subject"
    assert specs_from_config(None).keys() == DEFAULT_SPECS.keys()
    with pytest.raises(KeyError):
        specs_from_config({"nope": {}})
    with pytest.raises(KeyError):
        specs_from_config({"mmlu": {"stratify": "x"}})
