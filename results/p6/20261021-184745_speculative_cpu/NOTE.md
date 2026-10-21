# Token-level cascade, first CPU measurement (preliminary)

Lossless check passed: 24/24 generations identical to the full fp32 model (greedy).
No speed-up: median 0.87x (full-model int8 draft, 3.22 tokens per target forward), 0.63x (FLAP-20% int8, 2.57),
0.25x (FLAP-50% int8, 1.00 -- no draft token accepted). The int8 drafts' own cost per token was not measured
separately; the result implies roughly half of an fp32 target step (dynamic activation quantisation at batch 1).
Next: measure draft cost alone; shorter drafts (3 tokens) + draft confidence stop; SmolLM2-135M/360M drafts
(same tokenizer) as a reference.
