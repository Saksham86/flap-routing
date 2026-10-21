"""Faster execution of (pruned) models, chosen per device from measured bottlenecks.

CPU (edge, laptop): batch-1 decoding reads every weight once per token, so time ~ bytes read.
  * int8 dynamic quantization of all nn.Linear layers (weights stored as int8, activations quantized on the fly)
    -> 4x fewer weight bytes than fp32. Works for any layer shape, so it composes with FLAP's per-layer widths.
GPU (T4): batch-1 decoding in eager transformers is dominated by per-step launch overhead, not by weight size.
  * static KV cache + torch.compile(mode="reduce-overhead") -> the decode step is captured as a CUDA graph and
    replayed, removing Python/kernel-launch overhead. transformers' StaticLayer allocates each layer's cache from
    that layer's actual key/value shape, so per-layer pruned head counts are supported.
"""
from __future__ import annotations

import torch
from torch import nn


def quantize_int8_cpu(model: nn.Module) -> nn.Module:
    """Replace every nn.Linear (incl. lm_head) by an int8 dynamic-quantized Linear, in place."""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("int8 dynamic quantization is a CPU path")
    torch.backends.quantized.engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(model.float(), {nn.Linear}, dtype=torch.qint8, inplace=True)


def linear_weight_bytes(model: nn.Module) -> int:
    """Bytes of weights actually read per token by linear layers (float or int8-packed)."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
        elif hasattr(m, "weight") and callable(getattr(m, "weight", None)):   # quantized dynamic Linear
            try:
                w = m.weight()
                total += w.numel() * w.element_size()
            except Exception:
                pass
    return total


def enable_static_compile(model: nn.Module, max_cache_len: int | None = None) -> nn.Module:
    """GPU path: static KV cache. With a static (compileable) cache on CUDA, transformers' generate()
    auto-compiles the decode step with CompileConfig(mode="reduce-overhead") = CUDA graphs
    (transformers 5.17, GenerationMixin._valid_auto_compile_criteria). We do NOT wrap forward in torch.compile
    ourselves, which would compile twice. ``max_cache_len`` is unused (generate sizes the cache)."""
    gc = model.generation_config
    gc.cache_implementation = "static"
    gc.disable_compile = False
    return model
