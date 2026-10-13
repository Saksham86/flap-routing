"""Environment smoke test for the one transformers feature the project depends on most:
a Llama model whose layers have *different* numbers of attention heads and MLP channels
(what FLAP produces), with added output biases (FLAP's bias compensation).

The test builds a tiny random Llama, then compares
  (a) a "masked" copy: pruned heads/channels zeroed in place, compensation bias added, and
  (b) a "physical" copy: the same heads/channels actually removed (smaller matrices).
Both must give the same logits and the same greedy generation (with KV cache).
"""
from __future__ import annotations

import copy

import torch
from torch import nn


def _tiny_llama(seed: int, attn_impl: str):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        attention_bias=False, mlp_bias=False, tie_word_embeddings=True,
    )
    cfg._attn_implementation = attn_impl
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).eval()
    return model


def _with_bias(linear: nn.Linear, weight: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=True, dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        new.weight.copy_(weight)
        new.bias.copy_(bias)
    return new


def pruned_shape_smoke_test(device: str = "cpu", dtype: torch.dtype = torch.float32,
                            attn_impl: str = "sdpa", seed: int = 0) -> dict:
    base = _tiny_llama(seed, attn_impl).to(device=device, dtype=dtype)
    head_dim = base.config.hidden_size // base.config.num_attention_heads
    g = torch.Generator().manual_seed(seed + 1)
    # Different keep-sets per layer: (heads kept, MLP channels kept)
    plan = [([0, 1, 3], list(range(0, 96, 2))), ([2], list(range(10, 70))), ([0, 1, 2, 3], [5, 6, 7])]

    masked, physical = copy.deepcopy(base), copy.deepcopy(base)
    for li, (heads, chans) in enumerate(plan):
        rows = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads]).to(device)
        chans_t = torch.tensor(chans, device=device)
        attn_bias = (torch.randn(base.config.hidden_size, generator=g) * 0.05).to(device, dtype)
        mlp_bias = (torch.randn(base.config.hidden_size, generator=g) * 0.05).to(device, dtype)

        # (a) masked: zero pruned q/k/v rows and gate/up rows; o/down keep full shape + bias
        ma, mm = masked.model.layers[li].self_attn, masked.model.layers[li].mlp
        head_mask = torch.zeros(ma.q_proj.out_features, dtype=torch.bool, device=device)
        head_mask[rows] = True
        chan_mask = torch.zeros(mm.gate_proj.out_features, dtype=torch.bool, device=device)
        chan_mask[chans_t] = True
        with torch.no_grad():
            for lin in (ma.q_proj, ma.k_proj, ma.v_proj):
                lin.weight[~head_mask] = 0
            for lin in (mm.gate_proj, mm.up_proj):
                lin.weight[~chan_mask] = 0
        ma.o_proj = _with_bias(ma.o_proj, ma.o_proj.weight.data, attn_bias)
        mm.down_proj = _with_bias(mm.down_proj, mm.down_proj.weight.data, mlp_bias)

        # (b) physical: slice the matrices
        pa, pm = physical.model.layers[li].self_attn, physical.model.layers[li].mlp
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj"):
                old = getattr(pa, name)
                new = nn.Linear(old.in_features, len(rows), bias=False, dtype=dtype, device=device)
                new.weight.copy_(old.weight[rows])
                setattr(pa, name, new)
            pa.o_proj = _with_bias(pa.o_proj, pa.o_proj.weight.data[:, rows], attn_bias)
            for name in ("gate_proj", "up_proj"):
                old = getattr(pm, name)
                new = nn.Linear(old.in_features, len(chans), bias=False, dtype=dtype, device=device)
                new.weight.copy_(old.weight[chans_t])
                setattr(pm, name, new)
            pm.down_proj = _with_bias(pm.down_proj, pm.down_proj.weight.data[:, chans_t], mlp_bias)

    ids = torch.randint(0, base.config.vocab_size, (2, 12), generator=g).to(device)
    with torch.no_grad():
        lm = masked(ids).logits.float()
        lp = physical(ids).logits.float()
        gm = masked.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
        gp = physical.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    max_diff = (lm - lp).abs().max().item()
    n_params_masked = sum(p.numel() for p in masked.parameters())
    n_params_phys = sum(p.numel() for p in physical.parameters())
    logits_ok = max_diff <= tol
    gen_ok = bool(torch.equal(gm, gp))
    # In half precision the two models sum in a different order, so a greedy near-tie can
    # legitimately flip a token. There the generation check is informational; fp32 is strict.
    gen_required = dtype == torch.float32
    return {
        "device": device, "dtype": str(dtype), "attn_impl": attn_impl,
        "max_abs_logit_diff": max_diff, "tolerance": tol,
        "logits_match": logits_ok,
        "generation_match": gen_ok, "generation_required": gen_required,
        "params_masked": n_params_masked, "params_physical": n_params_phys,
        "passed": bool(logits_ok and (gen_ok or not gen_required) and n_params_phys < n_params_masked),
    }
