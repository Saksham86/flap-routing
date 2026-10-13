# STATUS

_Last updated: 2026-10-13_

## Phase overview
| Phase | State | Exit criterion met? | GPU-h used |
|---|---|---|---|
| P0 Setup | **Local work done; waiting for the Colab exit run** | No — needs `hello_gpu` PASSED on a Colab T4 with NVML readable | 0 |
| P1 FLAP port + ladder | Not started (blocked on P0) | — | 0 |
| P2 Query pool + outcomes | Not started | — | 0 |
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

### Remaining for the P0 exit
1. User creates a **private GitHub repo** and pushes this repo (commands in the chat).
2. User runs `notebooks/run_colab.ipynb` on a **T4** with the default P0 settings.
3. Exit = `hello.json` with `"passed": true` on Drive, `checks.nvml.ok = true`, and the counter support recorded. Then import the zip into `results/p0/`.

### What to look at in the Colab output
- `checks.nvml.counter_supported`: decides whether energy uses the NVML counter (preferred) or the sampler.
- `checks.matmul.energy.counter_joules` vs `sampled_joules`: should agree within ~10%. If they don't, the sampler interval or counter granularity needs attention before P1 profiling.
- `checks.pruned_shapes_*_float16`: the logit match must hold on CUDA fp16.

## Open issues / decisions
- **Edge device** for latency/energy profiling: not chosen yet. Default assumption: the user's laptop CPU (Intel, 16 logical cores). Needed by P1 (profiling), not by P0.
- Venues unconfirmed for RouterBench, RouteLLM, RAP, ConsRoute; cite the arXiv versions until checked (see `thesis/related_work_notes.md`).
- The proposal .docx still describes v1; the v2 rewrite is paused (pipeline diagram v2 already drawn).
- The local machine's global site-packages has a broken `faker` pytest plugin and an old `requests`/`urllib3` combination. Both are harmless to us but explain the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` flag.
