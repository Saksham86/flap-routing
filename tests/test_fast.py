"""int8 CPU path must work on FLAP-pruned models (per-layer widths) and stay close to fp32."""
import copy

import torch

from qacr.pruning.flap import apply_plan, flap_plan, flap_scores
from qacr.runtime.fast import linear_weight_bytes, quantize_int8_cpu


def _tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_int8_on_pruned_model_close_to_fp32():
    m = _tiny()
    g = torch.Generator().manual_seed(1)
    calib = [torch.randint(0, 97, (1, 16), generator=g) for _ in range(8)]
    pruned = apply_plan(copy.deepcopy(m), flap_plan(m, flap_scores(m, calib, "cpu", log=None), 0.4))
    ids = torch.randint(0, 97, (2, 20), generator=g)
    with torch.no_grad():
        ref = pruned(ids).logits
    fp32_bytes = linear_weight_bytes(pruned)
    q = quantize_int8_cpu(copy.deepcopy(pruned))
    with torch.no_grad():
        out = q(ids).logits
        gen = q.generate(ids[:1], max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert gen.shape[1] == 24                                            # generation works end to end
    assert linear_weight_bytes(q) < 0.3 * fp32_bytes                     # ~4x fewer weight bytes
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.99, float(cos)
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean()
    assert agree > 0.9, float(agree)
