import pandas as pd
import torch

from qacr.pruning.calib import format_question, pool_calibration


class CharTok:
    """Character-level fake tokenizer: lets the test decode windows back to text."""

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) for c in text]])
        return type("Enc", (), {"input_ids": ids})()


def _pool():
    rows = []
    for i in range(30):
        rows.append({"query_id": f"t/t/{i:05d}", "split": "train", "question_text": f"TRAINQ{i} what is x",
                     "context": None, "choices": ["alpha", "beta"], "gold": "1"})
    for i in range(30):
        rows.append({"query_id": f"t/t/{100 + i:05d}", "split": "test", "question_text": f"TESTQ{i} secret",
                     "context": None, "choices": ["gamma", "delta"], "gold": "0"})
    return pd.DataFrame(rows)


def test_only_train_questions_and_no_answers():
    calib, info = pool_calibration(CharTok(), _pool(), nsamples=50, seed=0, seqlen=40)
    assert info["n_questions"] == 30 and info["splits"] == ["train"]
    assert len(calib) == 50 and all(c.shape == (1, 40) for c in calib)
    text = "".join("".join(chr(int(t)) for t in c[0]) for c in calib)
    assert "TESTQ" not in text and "secret" not in text and "gamma" not in text   # no test leakage
    assert "TRAINQ" in text


def test_format_has_options_but_no_gold():
    s = format_question({"question_text": "Q?", "context": "ctx", "choices": ["a", "b"], "gold": "1"})
    assert s == "ctx\nQuestion: Q?\nA. a\nB. b\nAnswer:"
    assert format_question({"question_text": "2+2?", "context": None, "choices": None}) == "Question: 2+2?\nAnswer:"


def test_deterministic():
    a, _ = pool_calibration(CharTok(), _pool(), nsamples=10, seed=3, seqlen=20)
    b, _ = pool_calibration(CharTok(), _pool(), nsamples=10, seed=3, seqlen=20)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
