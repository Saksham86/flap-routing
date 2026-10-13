import math

import numpy as np
import pytest

from qacr.data.outcomes import OutcomeIntegrityError, aggregate_check, mc_scores, parse_task_samples


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _pool_row(doc_index, gold=1, metric="acc", choices=("aa", "b", "cccc"), h="h"):
    return {"query_id": f"t/t/{doc_index:05d}", "task": "t", "lm_task": "t", "gold": str(gold),
            "lm_choices": list(choices), "metric": metric, "lm_doc_hash": f"{h}{doc_index}"}


def _mc_sample(doc_id, lls, acc, acc_norm, h="h"):
    return {"doc_id": doc_id, "doc_hash": f"{h}{doc_id}", "filter": "none", "prompt_hash": "p",
            "arguments": [("ctx words here", " a")], "filtered_resps": [(ll, False) for ll in lls],
            "acc": acc, "acc_norm": acc_norm}


def test_mc_scores_matches_lm_eval_definition():
    lls = [-1.0, -0.5, -2.0]
    s = mc_scores(lls, ["aa", "b", "cccc"], gold=1)
    assert s["pred"] == 1 and s["acc"] == 1.0
    # acc_norm divides by char length: [-0.5, -0.5, -0.5] -> argmax first index (0)
    assert s["pred_norm"] == 0 and s["acc_norm"] == 0.0
    p = np.exp(lls) / np.exp(lls).sum()
    assert math.isclose(s["soft_acc"], p[1])
    with pytest.raises(OutcomeIntegrityError):
        mc_scores([-1.0], ["a", "b"], 0)


def test_parse_mc_ok_and_aggregate():
    pool = {3: _pool_row(3), 7: _pool_row(7, gold=0)}
    samples = [_mc_sample(3, [-1.0, -0.5, -2.0], 1.0, 0.0), _mc_sample(7, [-1.0, -0.5, -2.0], 0.0, 1.0)]
    rows = parse_task_samples(samples, pool, "v0", FakeTok(), "multiple_choice")
    assert [r["correct"] for r in rows] == [1, 0]
    assert rows[0]["n_ctx_tokens"] == 3
    chk = aggregate_check(rows, {"t": {"acc,none": 0.5}}, "t", "acc", None)
    assert chk["ours"] == 0.5
    with pytest.raises(OutcomeIntegrityError):
        aggregate_check(rows, {"t": {"acc,none": 0.6}}, "t", "acc", None)


def test_parse_detects_hash_mismatch():
    pool = {3: _pool_row(3)}
    bad = _mc_sample(3, [-1.0, -0.5, -2.0], 1.0, 0.0, h="other")
    with pytest.raises(OutcomeIntegrityError, match="doc_hash"):
        parse_task_samples([bad], pool, "v0", FakeTok(), "multiple_choice")


def test_parse_detects_metric_disagreement_and_missing_docs():
    pool = {3: _pool_row(3), 4: _pool_row(4)}
    wrong = _mc_sample(3, [-1.0, -0.5, -2.0], 0.0, 0.0)          # lm-eval says wrong, we say right
    with pytest.raises(OutcomeIntegrityError, match="recomputed acc"):
        parse_task_samples([wrong, _mc_sample(4, [-1.0, -0.5, -2.0], 1.0, 0.0)], pool, "v0", FakeTok(),
                           "multiple_choice")
    with pytest.raises(OutcomeIntegrityError, match="no logged sample"):
        parse_task_samples([_mc_sample(3, [-1.0, -0.5, -2.0], 1.0, 0.0)], pool, "v0", FakeTok(), "multiple_choice")
    with pytest.raises(OutcomeIntegrityError, match="not selected"):
        parse_task_samples([_mc_sample(9, [-1.0, -0.5, -2.0], 1.0, 0.0)], pool, "v0", FakeTok(), "multiple_choice")


def test_parse_generation_uses_configured_filter():
    row = {"query_id": "g/g/00001", "task": "g", "lm_task": "g", "gold": "18", "lm_choices": None,
           "metric": "exact_match", "lm_doc_hash": "h1"}

    def s(filt, em, extracted):
        return {"doc_id": 1, "doc_hash": "h1", "filter": filt, "prompt_hash": "p",
                "arguments": [("Q: a b c", {})], "resps": [["so the answer is 18 #### 18"]],
                "filtered_resps": [extracted], "exact_match": em}

    rows = parse_task_samples([s("strict-match", 0.0, "[invalid]"), s("flexible-extract", 1.0, "18")],
                              {1: row}, "v0", FakeTok(), "generate_until", "flexible-extract")
    assert rows[0]["correct"] == 1 and rows[0]["exact_match_strict"] == 0.0 and rows[0]["n_gen_tokens"] == 7
    with pytest.raises(OutcomeIntegrityError):
        parse_task_samples([s("strict-match", 0.0, "x")], {1: row}, "v0", FakeTok(), "generate_until",
                           "flexible-extract")
