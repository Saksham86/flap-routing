"""Load a model variant from a spec dict.

Spec keys:
  name   variant id used in outcome tables, e.g. "v0", "flap_p20"
  kind   "hf"   -> any Hugging Face model id or local directory (v0 = unpruned base)
         "flap" -> FLAP-pruned checkpoint (implemented in P1)
  path   model id or directory
  dtype  "float16" | "bfloat16" | "float32"   (T4 has no fast bf16: use float16 there)
"""
from __future__ import annotations

from typing import Any

import torch

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {name!r}") from None


def load_variant(spec: dict, device: str) -> tuple[Any, Any]:
    """Return (model in eval mode on ``device``, tokenizer)."""
    for k in ("name", "kind", "path"):
        if k not in spec:
            raise KeyError(f"variant spec missing '{k}': {spec}")
    dtype = parse_dtype(spec.get("dtype", "float16"))
    kind = spec["kind"]
    if kind == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["path"], revision=spec.get("revision"))
        model = AutoModelForCausalLM.from_pretrained(spec["path"], revision=spec.get("revision"), dtype=dtype)
    elif kind == "flap":
        raise NotImplementedError("FLAP checkpoints are produced and loaded from P1 onwards")
    else:
        raise ValueError(f"unknown variant kind {kind!r}")
    model.to(device).eval()
    return model, tok


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
