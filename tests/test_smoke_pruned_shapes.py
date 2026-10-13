"""The transformers build we pin must support per-layer pruned shapes + compensation bias."""
import pytest
import torch

from qacr.common.smoke import pruned_shape_smoke_test


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_pruned_shapes_cpu(attn_impl):
    r = pruned_shape_smoke_test(device="cpu", dtype=torch.float32, attn_impl=attn_impl, seed=0)
    assert r["logits_match"], r
    assert r["generation_match"], r
    assert r["params_physical"] < r["params_masked"]
    assert r["passed"]
