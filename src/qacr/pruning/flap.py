"""FLAP (An et al., AAAI 2024) ported to transformers 5.x Llama-architecture models, plus the
Wanda-SP and Magnitude-SP structured baselines from the same repository.

Ported from github.com/CASIA-IVA-Lab/FLAP (Apache-2.0): lib/prune.py, lib/layerwrapper.py,
lib/data.py, lib/eval.py. Behaviour is kept identical to the official code, including:
  * WIFV metric; attention scores are squared, MLP scores are not; per-layer standardisation
    (torch.std, unbiased); per-head score = mean over the head's channels.
  * AL-AM global threshold with an INTEGER compression weight for heads
    (official: torch.ones_like(indices) is int64, so 512/3 is stored as 170).
    Generalised from LLaMA-7B's head_dim=128 to int(4*head_dim/3) (SmolLM2: 85).
  * BiasGPT's running statistics, including the aliasing in the official code:
    ``old_baseline_inp = self.baseline_inp`` is updated in place, so the fluctuation update
    uses the *new* mean twice, and the first sample contributes 0.
  * Bias compensation from the fp16-cast baseline input.
  * Calibration on WikiText-2 train: random 128-token windows (seed 0), 1024 samples.
  * Perplexity on WikiText-2 test with non-overlapping 128-token windows.
Deviation (documented): Wanda-SP calibrates on WikiText-2 (128 samples), not C4, so all methods
see the same calibration text.

A pruning result is stored as a *plan* (keep masks + compensation biases), not as weights.
Applying a plan to the base model reproduces the pruned model; plans are a few hundred KB.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field

import torch
from torch import nn

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


# ---------------------------------------------------------------------------------------
# Data (lib/data.py get_wikitext2)
# ---------------------------------------------------------------------------------------
def wikitext2(tokenizer, nsamples: int, seed: int, seqlen: int, revision: str | None = None):
    from datasets import load_dataset
    train = load_dataset(*WIKITEXT, split="train", revision=revision)
    test = load_dataset(*WIKITEXT, split="test", revision=revision)
    trainenc = tokenizer(" ".join(train["text"]), return_tensors="pt").input_ids
    testenc = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    random.seed(seed)
    calib = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        calib.append(trainenc[:, i:i + seqlen])
    return calib, testenc


# ---------------------------------------------------------------------------------------
# Statistics (lib/layerwrapper.py)
# ---------------------------------------------------------------------------------------
class BiasStats:
    """Transcription of FLAP's BiasGPT for metric WIFV (aliasing behaviour preserved)."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        dev = layer.weight.device
        self.in_dim = layer.weight.shape[1]
        self.nsamples = 0
        self.baseline_inp = torch.zeros(self.in_dim, device=dev)
        self.fluc_inp = torch.zeros(self.in_dim, device=dev)

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()                 # (dim, tokens)
        old_baseline_inp = self.baseline_inp                     # alias, as in the official code
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.nsamples == 0:
            self.fluc_inp = 0
        else:
            self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
            self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) *
                                       (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)
        self.nsamples += batch_size


class NormStats:
    """Transcription of FLAP's WrappedGPT (used by Wanda-SP)."""

    def __init__(self, layer: nn.Linear):
        self.scaler_row = torch.zeros(layer.weight.shape[1], device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


# ---------------------------------------------------------------------------------------
# Layer-0 input capture (lib/prune.py prepare_calibration_input)
# ---------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


@torch.no_grad()
def capture_layer_inputs(model, calib: list[torch.Tensor], device) -> tuple[torch.Tensor, dict]:
    layers = model.model.layers
    dtype = next(model.parameters()).dtype
    seqlen = calib[0].shape[1]
    inps = torch.zeros((len(calib), seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache: dict = {"i": 0, "kwargs": None}
    original = layers[0]

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inps[cache["i"]] = hidden_states[0]
            cache["i"] += 1
            if cache["kwargs"] is None:
                cache["kwargs"] = {k: v for k, v in kwargs.items() if k in
                                   ("attention_mask", "position_ids", "position_embeddings")}
            raise _Stop

    layers[0] = Catcher(original)
    try:
        for ids in calib:
            try:
                model(ids.to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        layers[0] = original
    return inps, cache["kwargs"]


def _layer_out(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------------------
@dataclass
class PrunePlan:
    method: str
    ratio: float
    head_keep: torch.Tensor            # [L, H] bool
    chan_keep: torch.Tensor            # [L, I] bool
    attn_bias: torch.Tensor | None     # [L, hidden] (None = no compensation)
    mlp_bias: torch.Tensor | None      # [L, hidden]
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        from safetensors.torch import save_file
        t = {"head_keep": self.head_keep.to(torch.uint8).contiguous(),
             "chan_keep": self.chan_keep.to(torch.uint8).contiguous()}
        if self.attn_bias is not None:
            t["attn_bias"] = self.attn_bias.float().contiguous()
            t["mlp_bias"] = self.mlp_bias.float().contiguous()
        save_file(t, str(path), metadata={"method": self.method, "ratio": repr(self.ratio),
                                          "meta": json.dumps(self.meta, default=str)})

    @classmethod
    def load(cls, path) -> "PrunePlan":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            md = f.metadata()
            t = {k: f.get_tensor(k) for k in f.keys()}
        return cls(method=md["method"], ratio=float(md["ratio"]), head_keep=t["head_keep"].bool(),
                   chan_keep=t["chan_keep"].bool(), attn_bias=t.get("attn_bias"), mlp_bias=t.get("mlp_bias"),
                   meta=json.loads(md.get("meta", "{}")))


def _dims(model) -> tuple[int, int, int, int, int]:
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    if c.num_key_value_heads != c.num_attention_heads:
        raise NotImplementedError("this port supports multi-head attention only (n_kv_heads == n_heads)")
    if c.num_attention_heads * head_dim != c.hidden_size:
        raise NotImplementedError("expects n_heads * head_dim == hidden_size")
    return c.num_hidden_layers, c.num_attention_heads, head_dim, c.hidden_size, c.intermediate_size


def param_count(L: int, H: int, head_dim: int, hidden: int, I: int, head_keep, chan_keep) -> dict:
    """Prunable parameters (q,k,v,o + gate,up,down weights), before and after a plan."""
    full = L * (4 * H * head_dim * hidden + 3 * I * hidden)
    kept = int(head_keep.sum()) * 4 * head_dim * hidden + int(chan_keep.sum()) * 3 * hidden
    return {"prunable_full": full, "prunable_kept": kept, "achieved_ratio": 1 - kept / full}


def unpruned_plan(model) -> PrunePlan:
    L, H, _, _, I = _dims(model)
    return PrunePlan("none", 0.0, torch.ones(L, H, dtype=torch.bool), torch.ones(L, I, dtype=torch.bool), None, None)


# ---------------------------------------------------------------------------------------
# FLAP: scores (one calibration pass) -> plan for any ratio
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def flap_scores(model, calib: list[torch.Tensor], device, log=print) -> dict:
    """Run the official per-layer statistics on the UNPRUNED model (inputs of layer i+1 are the
    outputs of the unpruned layer i). Returns raw (unstandardised) metrics + baseline inputs."""
    L, H, head_dim, hidden, I = _dims(model)
    inps, kw = capture_layer_inputs(model, calib, device)
    outs = torch.zeros_like(inps)
    attn_metric, mlp_metric, attn_base, mlp_base = [], [], [], []
    for i, layer in enumerate(model.model.layers):
        o_proj, down_proj = layer.self_attn.o_proj, layer.mlp.down_proj
        stats = {"o": BiasStats(o_proj), "d": BiasStats(down_proj)}
        hooks = [o_proj.register_forward_hook(lambda m, inp, out: stats["o"].add_batch(inp[0].data)),
                 down_proj.register_forward_hook(lambda m, inp, out: stats["d"].add_batch(inp[0].data))]
        for j in range(len(calib)):
            outs[j] = _layer_out(layer, inps[j].unsqueeze(0), kw)[0]
        for h in hooks:
            h.remove()
        # WIFV: fluctuation * squared column norm of the output projection
        wo = stats["o"].fluc_inp * torch.sum(o_proj.weight.data.pow(2), dim=0)
        wd = stats["d"].fluc_inp * torch.sum(down_proj.weight.data.pow(2), dim=0)
        attn_metric.append((wo ** 2).float().cpu())
        mlp_metric.append(wd.float().cpu())
        attn_base.append(stats["o"].baseline_inp.half().cpu())
        mlp_base.append(stats["d"].baseline_inp.half().cpu())
        inps, outs = outs, inps
        if log and (i % 6 == 0 or i == L - 1):
            log(f"flap scores: layer {i + 1}/{L}")
    return {"attn_metric": torch.stack(attn_metric), "mlp_metric": torch.stack(mlp_metric),
            "attn_baseline": torch.stack(attn_base), "mlp_baseline": torch.stack(mlp_base),
            "head_dim": head_dim, "n_heads": H}


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)


def flap_masks(scores: dict, ratio: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """AL-AM global threshold (lib/prune.py), returns (head_keep [L,H], chan_keep [L,I], threshold)."""
    head_dim, H = scores["head_dim"], scores["n_heads"]
    L = scores["attn_metric"].shape[0]
    attn = _standardize(scores["attn_metric"]).reshape(L, -1, head_dim).mean(dim=2)
    mlp = _standardize(scores["mlp_metric"])
    if ratio <= 0:
        return torch.ones_like(attn, dtype=torch.bool), torch.ones_like(mlp, dtype=torch.bool), float("-inf")
    prune_metric = torch.cat([attn.view(-1), mlp.view(-1)])
    sorted_prune, indices = torch.sort(prune_metric, descending=True)
    compression_weight = torch.ones_like(indices)                     # int64, as in the official code
    compression_weight[indices < attn.numel()] = 4.0 * head_dim / 3   # truncated to int on assignment
    target = torch.sum(compression_weight) * (1 - ratio)
    threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - target))]
    return attn > threshold, mlp > threshold, float(threshold)


@torch.no_grad()
def compensation_bias(model, head_keep: torch.Tensor, chan_keep: torch.Tensor, scores: dict):
    """bias = (baseline_input * pruned_mask) @ W_out^T, computed in the model dtype as in FLAP."""
    head_dim = scores["head_dim"]
    attn_b, mlp_b = [], []
    for i, layer in enumerate(model.model.layers):
        wo, wd = layer.self_attn.o_proj.weight.data, layer.mlp.down_proj.weight.data
        pruned_ch = ~head_keep[i].repeat_interleave(head_dim)
        a = (scores["attn_baseline"][i].to(wo.device, wo.dtype) * pruned_ch.to(wo.device)) @ wo.T
        m = (scores["mlp_baseline"][i].to(wd.device, wd.dtype) * (~chan_keep[i]).to(wd.device)) @ wd.T
        attn_b.append(a.float().cpu())
        mlp_b.append(m.float().cpu())
    return torch.stack(attn_b), torch.stack(mlp_b)


def flap_plan(model, scores: dict, ratio: float, meta: dict | None = None) -> PrunePlan:
    head_keep, chan_keep, thr = flap_masks(scores, ratio)
    attn_b, mlp_b = compensation_bias(model, head_keep, chan_keep, scores)
    L, H, head_dim, hidden, I = _dims(model)
    return PrunePlan("flap", ratio, head_keep, chan_keep, attn_b, mlp_b,
                     meta={**(meta or {}), "threshold": thr,
                           **param_count(L, H, head_dim, hidden, I, head_keep, chan_keep)})


# ---------------------------------------------------------------------------------------
# Baselines: Wanda-SP (sequential, uses pruned outputs) and Magnitude-SP (data-free)
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def magnitude_sp_plan(model, ratio: float) -> PrunePlan:
    L, H, head_dim, hidden, I = _dims(model)
    hk, ck = [], []
    for layer in model.model.layers:
        w_o = torch.norm(layer.self_attn.o_proj.weight.data.float(), dim=0).reshape(-1, head_dim).sum(dim=1)
        w_d = torch.norm(layer.mlp.down_proj.weight.data.float(), dim=0)
        hk.append((w_o >= torch.sort(w_o)[0][int(ratio * H)]).cpu())
        ck.append((w_d >= torch.sort(w_d)[0][int(w_d.numel() * ratio)]).cpu())
    hk, ck = torch.stack(hk), torch.stack(ck)
    return PrunePlan("mag_sp", ratio, hk, ck, None, None, meta=param_count(L, H, head_dim, hidden, I, hk, ck))


@torch.no_grad()
def wanda_sp_plan(model, calib: list[torch.Tensor], ratio: float, device, log=print) -> PrunePlan:
    """Runs on a deep copy, pruning (masking) layer by layer; the base model is left untouched."""
    L, H, head_dim, hidden, I = _dims(model)
    work = copy.deepcopy(model)
    inps, kw = capture_layer_inputs(work, calib, device)
    outs = torch.zeros_like(inps)
    hk, ck = [], []
    for i, layer in enumerate(work.model.layers):
        o_proj, down_proj = layer.self_attn.o_proj, layer.mlp.down_proj
        stats = {"o": NormStats(o_proj), "d": NormStats(down_proj)}
        hooks = [o_proj.register_forward_hook(lambda m, inp, out: stats["o"].add_batch(inp[0].data)),
                 down_proj.register_forward_hook(lambda m, inp, out: stats["d"].add_batch(inp[0].data))]
        for j in range(len(calib)):
            outs[j] = _layer_out(layer, inps[j].unsqueeze(0), kw)[0]
        for h in hooks:
            h.remove()
        m_o = (torch.abs(o_proj.weight.data) * torch.sqrt(stats["o"].scaler_row.reshape(1, -1))).mean(dim=0)
        m_o = m_o.reshape(-1, head_dim).sum(dim=1)
        keep_h = m_o >= torch.sort(m_o)[0][int(ratio * H)]
        m_d = (torch.abs(down_proj.weight.data) * torch.sqrt(stats["d"].scaler_row.reshape(1, -1))).mean(dim=0)
        keep_c = m_d >= torch.sort(m_d)[0][int(m_d.numel() * ratio)]
        _mask_layer(layer, keep_h, keep_c, head_dim, None, None)
        for j in range(len(calib)):                                      # pruned outputs feed the next layer
            outs[j] = _layer_out(layer, inps[j].unsqueeze(0), kw)[0]
        inps, outs = outs, inps
        hk.append(keep_h.cpu())
        ck.append(keep_c.cpu())
        if log and (i % 6 == 0 or i == L - 1):
            log(f"wanda-sp {ratio}: layer {i + 1}/{L}")
    del work
    hk, ck = torch.stack(hk), torch.stack(ck)
    return PrunePlan("wanda_sp", ratio, hk, ck, None, None, meta=param_count(L, H, head_dim, hidden, I, hk, ck))


# ---------------------------------------------------------------------------------------
# Applying plans
# ---------------------------------------------------------------------------------------
def _linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias.to(weight.device, weight.dtype))
    return lin


@torch.no_grad()
def _mask_layer(layer, keep_h, keep_c, head_dim, attn_bias, mlp_bias) -> None:
    """In-place masking (shapes unchanged): zero pruned q/k/v rows and gate/up rows."""
    rows = keep_h.repeat_interleave(head_dim).to(layer.self_attn.q_proj.weight.device)
    for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
        lin.weight.data[~rows] = 0
    ch = keep_c.to(layer.mlp.gate_proj.weight.device)
    for lin in (layer.mlp.gate_proj, layer.mlp.up_proj):
        lin.weight.data[~ch] = 0
    if attn_bias is not None:
        layer.self_attn.o_proj = _linear(layer.self_attn.o_proj.weight.data, attn_bias)
        layer.mlp.down_proj = _linear(layer.mlp.down_proj.weight.data, mlp_bias)


@torch.no_grad()
def apply_plan(model, plan: PrunePlan, physical: bool = True):
    """Prune ``model`` in place according to ``plan`` and return it.
    physical=True removes rows/columns (smaller matrices); False only masks (for testing)."""
    L, H, head_dim, hidden, I = _dims(model)
    if plan.head_keep.shape != (L, H) or plan.chan_keep.shape != (L, I):
        raise ValueError("plan shape does not match model")
    empty = [i for i in range(L) if not plan.head_keep[i].any() or not plan.chan_keep[i].any()]
    if empty and physical:
        raise NotImplementedError(f"plan removes ALL heads or ALL channels in layers {empty}; "
                                  "not supported by this runtime")
    for i, layer in enumerate(model.model.layers):
        ab = None if plan.attn_bias is None else plan.attn_bias[i]
        mb = None if plan.mlp_bias is None else plan.mlp_bias[i]
        if not physical:
            _mask_layer(layer, plan.head_keep[i], plan.chan_keep[i], head_dim, ab, mb)
            continue
        att, mlp = layer.self_attn, layer.mlp
        dev = att.q_proj.weight.device
        rows = torch.where(plan.head_keep[i].repeat_interleave(head_dim))[0].to(dev)
        att.q_proj = _linear(att.q_proj.weight.data[rows], None)
        att.k_proj = _linear(att.k_proj.weight.data[rows], None)
        att.v_proj = _linear(att.v_proj.weight.data[rows], None)
        att.o_proj = _linear(att.o_proj.weight.data[:, rows].contiguous(), ab)
        ch = torch.where(plan.chan_keep[i])[0].to(dev)
        mlp.gate_proj = _linear(mlp.gate_proj.weight.data[ch], None)
        mlp.up_proj = _linear(mlp.up_proj.weight.data[ch], None)
        mlp.down_proj = _linear(mlp.down_proj.weight.data[:, ch].contiguous(), mb)
    return model


# ---------------------------------------------------------------------------------------
# Perplexity (lib/eval.py eval_ppl_wikitext, batched; identical value)
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, testenc: torch.Tensor, seqlen: int = 128, bs: int = 16, device=None,
             max_windows: int | None = None) -> float:
    device = device or next(model.parameters()).device
    nsamples = testenc.numel() // seqlen
    if max_windows is not None:
        nsamples = min(nsamples, max_windows)
    nlls = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    for i in range(0, nsamples, bs):
        j = min(i + bs, nsamples)
        inputs = testenc[:, i * seqlen:j * seqlen].to(device).reshape(j - i, seqlen)
        logits = model(inputs, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = inputs[:, 1:]
        # per-window mean loss * seqlen, exactly what the official bs=1 loop accumulates
        per_tok = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(per_tok.view(j - i, -1).mean(dim=1) * seqlen)
    return float(torch.exp(torch.cat(nlls).sum() / (nsamples * seqlen)))
