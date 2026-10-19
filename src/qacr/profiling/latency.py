"""Batch-1 latency / energy / memory profiling of one model on one device.

Measured quantities (median over ``reps`` after ``warmup`` runs, synchronised on CUDA):
  prefill_ms[L]      one forward pass over an L-token prompt (no cache reuse)
  gen_ms             generate exactly N new tokens (greedy, EOS suppressed) after a 128-token prompt
  decode_ms_per_tok  (gen_ms - prefill_ms[128]) / (N - 1)   -- HF generate: 1 prefill + N-1 decode steps
  energy             NVML energy for the whole generation repetitions window (GPU only)
Prompts are random token ids: latency depends on shapes, not on content.
"""
from __future__ import annotations

import statistics
import time

import torch


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def _time(fn, device: str, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


@torch.no_grad()
def profile_model(model, device: str, vocab_size: int, prompt_lens=(128, 512, 1024), gen_tokens: int = 64,
                  warmup: int = 3, reps: int = 10, seed: int = 0, energy_meter=None) -> dict:
    g = torch.Generator().manual_seed(seed)
    res: dict = {"device": device, "prompt_lens": list(prompt_lens), "gen_tokens": gen_tokens,
                 "warmup": warmup, "reps": reps}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    prefill = {}
    for L in prompt_lens:
        ids = torch.randint(0, vocab_size, (1, L), generator=g).to(device)
        ts = _time(lambda: model(ids, use_cache=False), device, warmup, reps)
        prefill[L] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts)}
    res["prefill"] = prefill

    base_len = 128
    ids = torch.randint(0, vocab_size, (1, base_len), generator=g).to(device)
    eos = getattr(model.generation_config, "eos_token_id", None)

    def gen():
        model.generate(ids, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens, do_sample=False,
                       pad_token_id=eos if isinstance(eos, int) else 0)

    for _ in range(warmup):
        gen()
    _sync(device)
    if energy_meter is not None and energy_meter.available:
        energy_meter.start()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gen()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    energy = energy_meter.stop().to_dict() if (energy_meter is not None and energy_meter.available) else None
    gen_ms = statistics.median(ts)
    pre128 = prefill.get(base_len, {}).get("median_ms")
    res["generate"] = {"prompt_len": base_len, "median_ms": gen_ms, "min_ms": min(ts), "max_ms": max(ts),
                       "decode_ms_per_tok": (gen_ms - pre128) / (gen_tokens - 1) if pre128 is not None else None,
                       "tokens_per_s": gen_tokens / (gen_ms / 1000.0)}
    if energy is not None:
        n_tok = reps * gen_tokens
        j = energy.get("counter_joules") if energy.get("counter_joules") is not None else energy.get("sampled_joules")
        energy["joules_per_generated_token"] = (j / n_tok) if j is not None else None
    res["energy"] = energy
    if device.startswith("cuda"):
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    res["weight_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    res["n_params"] = sum(p.numel() for p in model.parameters())
    return res
