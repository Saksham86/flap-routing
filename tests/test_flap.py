"""Tests for the FLAP port on a tiny random Llama (CPU, fp32)."""
import copy

import pytest
import torch
from torch import nn

from qacr.pruning.flap import (BiasStats, PrunePlan, apply_plan, eval_ppl, flap_masks, flap_plan, flap_scores,
                               magnitude_sp_plan, param_count, unpruned_plan, wanda_sp_plan)


def tiny(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).eval()


def calib(n=12, seqlen=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, seqlen), generator=g) for _ in range(n)]


def test_bias_stats_matches_official_recurrence():
    """Independent re-derivation of BiasGPT's update, INCLUDING the aliasing of
    old_baseline_inp (which makes both factors use the updated mean) and fluc=0 after sample 0."""
    torch.manual_seed(0)
    xs = [torch.randn(1, 7, 5) for _ in range(6)]
    s = BiasStats(nn.Linear(5, 3))
    for x in xs:
        s.add_batch(x)
    b = torch.zeros(5)
    fluc = None
    for t, x in enumerate(xs):
        xt = x.reshape(-1, 5).t()
        b = b * t / (t + 1) + xt.mean(dim=1) / (t + 1)
        if t == 0:
            fluc = torch.zeros(5)
        else:
            fluc = fluc * (t - 1) / t + ((xt - b.unsqueeze(1)) ** 2).sum(dim=1) / (t + 1)
    assert torch.allclose(s.baseline_inp, b, atol=1e-6)
    assert torch.allclose(s.fluc_inp, fluc, atol=1e-5)


def test_ratio_zero_is_identity():
    m = tiny()
    ids = calib(1)[0]
    ref = m(ids).logits
    scores = flap_scores(m, calib(), "cpu", log=None)
    plan = flap_plan(m, scores, 0.0)
    assert plan.head_keep.all() and plan.chan_keep.all()
    assert torch.count_nonzero(plan.attn_bias) == 0 and torch.count_nonzero(plan.mlp_bias) == 0
    out = apply_plan(copy.deepcopy(m), plan)(ids).logits
    assert torch.allclose(out, ref, atol=1e-5)


def test_scores_do_not_modify_model():
    m = tiny()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    flap_scores(m, calib(), "cpu", log=None)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


@pytest.mark.parametrize("ratio", [0.2, 0.4])
def test_masked_equals_physical_and_smaller(ratio):
    m = tiny()
    scores = flap_scores(m, calib(), "cpu", log=None)
    plan = flap_plan(m, scores, ratio)
    ids = calib(2, seed=9)[0]
    masked = apply_plan(copy.deepcopy(m), plan, physical=False)
    phys = apply_plan(copy.deepcopy(m), plan, physical=True)
    assert torch.allclose(masked(ids).logits, phys(ids).logits, atol=1e-4)
    g1 = masked.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
    g2 = phys.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
    assert torch.equal(g1, g2)
    assert sum(p.numel() for p in phys.parameters()) < sum(p.numel() for p in m.parameters())


def test_bias_is_expected_contribution_of_pruned_channels():
    m = tiny()
    scores = flap_scores(m, calib(), "cpu", log=None)
    plan = flap_plan(m, scores, 0.4)
    i = 0
    W = m.model.layers[i].mlp.down_proj.weight.data
    pruned = ~plan.chan_keep[i]
    expected = W[:, pruned] @ scores["mlp_baseline"][i].float()[pruned]
    assert torch.allclose(plan.mlp_bias[i], expected, atol=1e-3)


def test_masks_are_nested_across_ratios():
    m = tiny()
    scores = flap_scores(m, calib(), "cpu", log=None)
    prev_h, prev_c = None, None
    for r in (0.1, 0.2, 0.35, 0.5):
        h, c, _ = flap_masks(scores, r)
        if prev_h is not None:
            assert not (h & ~prev_h).any() and not (c & ~prev_c).any()   # kept(r) subset of kept(r-)
        prev_h, prev_c = h, c


def test_achieved_ratio_at_smollm2_dims():
    """Global AL-AM threshold hits the requested ratio at the real SmolLM2-1.7B sizes."""
    L, H, hd, hidden, I = 24, 32, 64, 2048, 8192
    g = torch.Generator().manual_seed(0)
    scores = {"attn_metric": torch.rand(L, hidden, generator=g) ** 2, "mlp_metric": torch.rand(L, I, generator=g),
              "head_dim": hd, "n_heads": H}
    for r in (0.2, 0.35, 0.5):
        h, c, _ = flap_masks(scores, r)
        got = param_count(L, H, hd, hidden, I, h, c)["achieved_ratio"]
        assert abs(got - r) < 0.01, (r, got)


def test_plan_roundtrip(tmp_path):
    m = tiny()
    plan = flap_plan(m, flap_scores(m, calib(), "cpu", log=None), 0.3, meta={"x": 1})
    p = tmp_path / "plan.safetensors"
    plan.save(p)
    q = PrunePlan.load(p)
    assert q.method == "flap" and q.ratio == 0.3 and q.meta["x"] == 1
    assert torch.equal(q.head_keep, plan.head_keep) and torch.equal(q.chan_keep, plan.chan_keep)
    assert torch.equal(q.attn_bias, plan.attn_bias.float()) and torch.equal(q.mlp_bias, plan.mlp_bias.float())


def test_baselines_prune_uniformly():
    m = tiny()
    for plan in (magnitude_sp_plan(m, 0.5), wanda_sp_plan(m, calib(), 0.5, "cpu", log=None)):
        assert (plan.head_keep.sum(dim=1) == 2).all()            # 4 heads, int(0.5*4)=2 pruned per layer
        assert (plan.chan_keep.sum(dim=1) == 48).all()           # 96 channels, 48 pruned per layer
        assert plan.attn_bias is None
        apply_plan(copy.deepcopy(m), plan)                         # applies without error
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    wanda_sp_plan(m, calib(), 0.5, "cpu", log=None)
    assert all(torch.equal(w0[k], v) for k, v in m.state_dict().items())   # base untouched


def test_unpruned_plan_and_empty_layer_guard():
    m = tiny()
    plan = unpruned_plan(m)
    plan.head_keep[1] = False
    with pytest.raises(NotImplementedError):
        apply_plan(copy.deepcopy(m), plan)


def test_eval_ppl_matches_official_bs1_loop():
    m = tiny()
    g = torch.Generator().manual_seed(3)
    testenc = torch.randint(0, 97, (1, 16 * 7 + 5), generator=g)
    ours = eval_ppl(m, testenc, seqlen=16, bs=3)
    nlls = []
    with torch.no_grad():                                          # transcription of lib/eval.py
        for i in range(7):
            inputs = testenc[:, i * 16:(i + 1) * 16]
            lm_logits = m(inputs).logits
            loss = nn.CrossEntropyLoss()(lm_logits[:, :-1, :].reshape(-1, 97), inputs[:, 1:].reshape(-1))
            nlls.append(loss.float() * 16)
    ref = torch.exp(torch.stack(nlls).sum() / (7 * 16)).item()
    assert abs(ours - ref) / ref < 1e-5
