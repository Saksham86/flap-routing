# STATUS

_Last updated: 2026-10-13_

## Phase overview
| Phase | State | Exit criterion met? | GPU-h used |
|---|---|---|---|
| P0 Setup | **Done** (Colab exit run passed 2026-10-19) | **Yes** — `hello_gpu` passed on a Tesla T4, NVML counter supported | ~0.05 |
| P1 FLAP port + ladder | Not started (blocked on P0) | — | 0 |
| P2 Query pool + outcomes | **CPU parts done on branch `p2-query-pool`**; GPU outcome runs pending (v0 after the P0 exit; v1–v3 after P1) | No — needs outcome runs for all 4 variants + headroom report | 0 |
| P3 Routers | Not started | — | 0 |
| P4 Runtime + pipeline | Not started | — | 0 |
| P5 Evaluation | Not started | — | 0 |
| P6 Cascade (optional) | Not started | — | 0 |
| P7 Ablations + thesis | Not started | — | 0 |

**GPU-hours total: 0** (budget ~20–30)

## P0 — Setup

### Done (verified locally: Windows 11, Python 3.12.3, torch 2.7.0 CPU)
- Repo scaffold, `pyproject.toml`, package `qacr`.
- Pinned `requirements.txt`:
  - Every version is ≥ 2 weeks old.
  - Installed together in a clean venv; `pip check` reports no conflicts among them.
  - torch is intentionally unpinned, so Colab's CUDA build is used.
- Utilities:
  - config loader with `--set` overrides and `${ENV}` expansion;
  - seeding;
  - run directories (config/meta/env/pip-freeze snapshot, atomic JSON writes, safe resume);
  - logging;
  - NVML energy meter (counter + sampler, degrades gracefully without a GPU).
- Pruned-shape smoke test:
  - Confirms transformers 5.17 runs Llama layers with different head/channel counts per layer plus compensation biases.
  - The masked and physically pruned models give identical logits (max diff 9e-8) and identical greedy generation, for both eager and sdpa attention.
  - A deliberately broken bias makes it fail, so the test is not vacuous.
- `scripts/p0_hello_gpu.py`:
  - Ran end to end on the local CPU and PASSED.
  - The Hub check re-confirmed the SmolLM2-1.7B config: 24 layers, 32/32 heads, hidden 2048, MLP 8192, no bias, tied embeddings.
- `scripts/import_results.py` brings Colab result zips into `results/`, with zip-slip protection and no silent overwrite.
- `notebooks/run_colab.ipynb`: mount Drive → clone/pull (private repo via the `GH_TOKEN` Colab secret) → install → **pytest** → run → summary + zip download.
- Tests: **12/12 pass** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`).
- `thesis/related_work_notes.md`: 18 references, metadata verified via the arXiv API and the PDFs' DOIs.

### P0 exit run (Colab, Tesla T4) — run `results/p0/20261019-144313_hello_gpu` (values as printed by the notebook; zip to be imported)
- All checks passed: seeding reproducible; pruned-shape smoke test passed on CUDA in fp32 and fp16, eager and sdpa (fp16 max logit diff 0.0); Hub config matches the plan.
- NVML energy counter supported. It agrees with the 20 ms power sampler to within 1.7% (idle: 30.21 J vs 30.72 J over 3.0 s) and 1.1% (fp16 matmul load: 692.4 J vs 700.0 J over 10.0 s).
- Idle power 10.2 W; matmul load mean 69.7 W (max 82.2 W); fp16 4096x4096 matmul: 23.6 TFLOPS.
- Colab runtime: Python 3.13.

### Remaining for the P0 exit (historical)
1. User creates a **private GitHub repo** and pushes this repo (commands in the chat).
2. User runs `notebooks/run_colab.ipynb` on a **T4** with the default P0 settings.
3. Exit = `hello.json` with `"passed": true` on Drive, `checks.nvml.ok = true`, and the counter support recorded. Then import the zip into `results/p0/`.

### What to look at in the Colab output
- `checks.nvml.counter_supported`: decides whether energy uses the NVML counter (preferred) or the sampler.
- `checks.matmul.energy.counter_joules` vs `sampled_joules`: should agree within ~10%. If they don't, the sampler interval or counter granularity needs attention before P1 profiling.
- `checks.pruned_shapes_*_float16`: the logit match must hold on CUDA fp16.

## P2 — Query pool + outcome matrix (branch `p2-query-pool`)

### Done (CPU, verified locally)
- **Query pool** `data/p2/query_pool.parquet`: **19,242 queries**, seed 2026, built from lm-eval 0.4.13's own task objects. Run record: `results/p2/20261013-142932_build_pool/` (clean commit).

  | task | eval docs | kept | train / val / test | primary metric | shots |
  |---|---|---|---|---|---|
  | arc_easy | 2376 | 2376 | 1426 / 475 / 475 | acc | 0 |
  | arc_challenge | 1172 | 1172 | 703 / 234 / 235 | acc_norm | 0 |
  | hellaswag | 10042 | 4000 (cap) | 2400 / 800 / 800 | acc_norm | 0 |
  | winogrande | 1267 | 1267 | 760 / 253 / 254 | acc | 0 |
  | piqa | 1838 | 1838 | 1103 / 368 / 367 | acc | 0 |
  | boolq | 3270 | 3270 | 1962 / 654 / 654 | acc | 0 |
  | mmlu | 14042 (105 exact dupes dropped) | 4000 (cap, by subject) | 2397 / 803 / 800 | acc | 0 |
  | gsm8k | 1319 | 1319 | 791 / 264 / 264 | exact match, flexible-extract | 5 |
  | **total** | | **19,242** | **11,542 / 3,851 / 3,849** | | |

  - Splits are stratified per task (and per MMLU subject).
  - Splits are deterministic: rebuilt twice, identical, and a task's split does not depend on which other tasks are built.
  - Each row stores lm-eval's doc hash. The dataset Hub commit shas are in `manifest.json`, and publishing refuses if any is missing.
- **Outcome pipeline** `scripts/p2_outcomes.py` (one variant per run, resumable per task). Verified end to end on CPU with SmolLM2-135M:
  - every per-sample check passes (doc hash; recomputed acc/acc_norm equal lm-eval's; per-task mean equals lm-eval's aggregate);
  - two runs are **bit-identical**: same prompts, including the GSM8K 5-shot exemplars, same outcomes, log-likelihood diff 0.0.
- **RouterBench external labels** `data/p2/routerbench_labels.parquet`: **10,420 queries**.

  | task | matched |
  |---|---|
  | HellaSwag | 4000/4000 |
  | Winogrande | 1267/1267 |
  | ARC-C | 1172/1172 |
  | MMLU | 3981/4000 (19 ambiguous duplicates left unlabelled) |
  | GSM8K | 0 — RouterBench used the GSM8K **train** split (7,450/7,473 train questions found, 0/1,319 test) |
  | ARC-E, PIQA, BoolQ | none — not in RouterBench |

  The pickle is loaded only through a whitelisted unpickler (11 pandas/numpy globals), pinned to revision `78402148` and SHA-256 `ba4f77f1…`.
- Tests: **34 passed** (+1 opt-in real-file test, skipped unless QACR_ROUTERBENCH_PKL is set).

### Problems found and fixed during P2 (each now has a test)
1. lm-eval 0.4.13 maps `samples=` results back by position, which is only correct for **sorted** indices. The script enforces sorting, and the parser cross-checks doc hashes.
2. RouterBench **shuffles answer options** relative to the source datasets. Order-sensitive matching had silently matched only ~4% (Winogrande exactly 50%). Keys are now order-invariant.
3. RouterBench GSM8K prompts are `[exemplars, question]` lists, which caused 7,450 parse failures. Fixed.
4. Run provenance recorded `dirty: true` for every run made inside the repo, because `results/` was counted. Fixed; the fix needs to reach `main` too (see below).
5. Mixed column types (gold int vs str) broke the combined parquet. Outcomes now use one fixed schema.

### Decisions made in P2 — please confirm (they define "correct")
- Multiple-choice tasks are **0-shot**, as in the FLAP / LLM-Pruner papers.
- Primary metric: **acc_norm** for HellaSwag and ARC-Challenge, **acc** elsewhere. Both are stored.
- GSM8K: **5-shot** (lm-eval default). Correct = **flexible-extract** exact match, which is more robust to the format drift pruning may cause; strict-match is stored too. Generation is capped at **256 tokens**.
- Caps: MMLU and HellaSwag **4000** each.

### Next (GPU, Colab — after the P0 exit passes)
- **v0 outcome run.** Notebook settings:
  - `BRANCH = "p2-query-pool"`
  - `SCRIPT = "scripts/p2_outcomes.py"`
  - `CONFIG = "configs/p2_outcomes.yaml"`
  - `EXTRA = "--set require_gpu=true"`
- Expected load: ~59k log-likelihood requests plus 1,319 GSM8K generations. My guess is ~30–60 min on a T4; the run will measure it.
- Sanity check: v0 scores should land near SmolLM2-1.7B's published base scores.
- v1–v3 run the same way once P1 produces FLAP checkpoints (`variant.kind: flap`). Then build the outcome matrix and the **headroom report** (decision gate).

## Open issues / decisions
- **Edge device** for latency/energy profiling: not chosen yet. Default assumption: the user's laptop CPU (Intel, 16 logical cores). Needed by P1 (profiling), not by P0.
- Venues unconfirmed for RouterBench, RouteLLM, RAP, ConsRoute; cite the arXiv versions until checked (see `thesis/related_work_notes.md`).
- The proposal .docx still describes v1; the v2 rewrite is paused (pipeline diagram v2 already drawn).
- The local machine's global site-packages has a broken `faker` pytest plugin and an old `requests`/`urllib3` combination. Both are harmless to us but explain the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` flag.
