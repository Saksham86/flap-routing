import io
import os
import pickle

import pandas as pd
import pytest

from qacr.data.routerbench import (ALLOWED_GLOBALS, ROUTERBENCH_0SHOT_SHA256, UnsafePickleError,
                                   _RestrictedUnpickler, match_key, match_pool, parse_gsm8k_question,
                                   parse_mc_prompt, prompt_text, safe_load_pickle)


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_blocks_code():
    payload = pickle.dumps(_Evil())
    with pytest.raises(UnsafePickleError):
        _RestrictedUnpickler(io.BytesIO(payload)).load()


def test_whitelist_is_only_data_reconstruction():
    allowed_modules = {"builtins", "numpy", "numpy.core.multiarray", "numpy.core.numeric",
                       "pandas._libs.internals", "pandas.core.frame", "pandas.core.indexes.base",
                       "pandas.core.indexes.range", "pandas.core.internals.managers"}
    assert {m for m, _ in ALLOWED_GLOBALS} <= allowed_modules
    assert ("builtins", "eval") not in ALLOWED_GLOBALS and ("builtins", "exec") not in ALLOWED_GLOBALS
    assert [n for m, n in ALLOWED_GLOBALS if m == "builtins"] == ["slice"]


@pytest.mark.skipif(not os.environ.get("QACR_ROUTERBENCH_PKL"), reason="set QACR_ROUTERBENCH_PKL to run")
def test_real_routerbench_file_loads_safely():
    df = safe_load_pickle(os.environ["QACR_ROUTERBENCH_PKL"], ROUTERBENCH_0SHOT_SHA256)
    assert df.shape == (36497, 37) and "eval_name" in df.columns


def test_parse_mc_prompt():
    text = ("Please answer with the letter of the correct answer.\n\nWhat is 2+2?\nA) 3\nB) 4\nC) 5\n"
            'Print only a single choice from "A" or "B" or "C" without explanation. Answer:')
    stem, opts = parse_mc_prompt(text)
    assert stem == "What is 2+2?" and opts == ["3", "4", "5"]
    assert prompt_text("['hello\\nworld']") == "hello\nworld"


def test_parse_gsm8k_takes_last_question():
    t = "TASK...\nQuestion: ex1?\nAnswer: 1\n\nQuestion: real one?\nAnswer:"
    assert parse_gsm8k_question(t) == "real one?"


def test_match_pool_and_ambiguity():
    rb = pd.DataFrame({
        "sample_id": ["w.0", "w.1", "w.2"],
        "prompt": [str(["Tom _ ran.\nA) x\nB) y\nPrint only"]), str(["Amb _.\nA) p\nB) q\nPrint"]),
                   str(["Amb _.\nA) p\nB) q\nPrint"])],
        "eval_name": ["winogrande"] * 3,
        "m1": [1.0, 1.0, 0.0], "m2": [0.0, 1.0, 0.0],
        "m1|model_response": ["", "", ""],
    })
    pool = pd.DataFrame({"query_id": ["winogrande/winogrande/00000", "winogrande/winogrande/00001",
                                      "piqa/piqa/00000"],
                         "task": ["winogrande", "winogrande", "piqa"],
                         "question_text": ["Tom _ ran.", "Amb _.", "goal"],
                         "choices": [["x", "y"], ["p", "q"], ["a", "b"]]})
    labels, rep = match_pool(pool, rb)
    assert list(labels["query_id"]) == ["winogrande/winogrande/00000"]
    assert labels["rb_mean_score"].iloc[0] == 0.5
    assert rep["winogrande"]["ambiguous"] == 1 and rep["piqa"]["matched"] == 0


def test_hellaswag_key_uses_lm_eval_preprocessing():
    raw = match_key("hellaswag", "ignored", ["Do it [title] now", "b"])
    assert raw == match_key("hellaswag", "different stem", ["Do it. now", "b"])
