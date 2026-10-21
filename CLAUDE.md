# CLAUDE.md — Query-Adaptive Compression-Level Routing (B.Tech Thesis) — v2

> v1 is preserved in `CLAUDE_PLAN_v1.md`. The topic is unchanged. v2 changes the base model, the data/evaluation engine, the router formulation and the deployment design so that the claims the thesis makes are actually measurable. Items marked **[APPROVE]** need the user's sign-off before the phase that uses them.

## Goal
Use FLAP to build a ladder of compression levels of **one** small LLM, train a cheap router that picks a level **per query**, and show — against static, task-level and oracle baselines — how much latency/energy can be saved at (near-)equal accuracy on edge-class hardware. Deliverables: working pipeline, accuracy–cost curves, thesis.

## Research questions (the thesis is structured around these)
- **RQ1 — Ladder.** Does FLAP give a useful accuracy–cost ladder on a ~1.7B edge model, and does the paper's ranking (FLAP > Wanda-SP > Magnitude-SP) hold at this scale?
- **RQ2 — Headroom.** How much can *per-query* choice of compression level gain over the best *fixed* level? (Measured with an oracle before any router is built. If headroom is ~0, routing cannot help, and that is the finding.)
- **RQ3 — Routing.** How much of that headroom does a cheap router recover, and does it beat routing on **task type alone** (i.e. is it really reading query content)?
- **RQ4 — System cost.** What are the end-to-end savings in latency, energy and memory under realistic deployment modes, with router overhead included?

## Positioning (closest prior work — must be cited and differentiated)
- **PuDDing** (Wee, Park, Lee — ICML 2025): a lightweight router picks one of several depth-pruned variants per prompt. It chooses *which* blocks to skip at a fixed budget.
- **Instruction-Following Pruning** (Apple, arXiv 2501.02086): per-prompt structured width masks. Requires training.
- **Flextron** (Cai et al. — ICML 2024): a nested elastic LLM with an input-adaptive router. Requires training/distillation.
- **MatFormer**, **RouteLLM / RouterBench**, **RAP**, **ConsRoute / HybridFlow**: verify exact citations in P0.

**Our claim (narrow):**
- A *training-free* (FLAP) ladder of nested width-pruned variants of one model.
- Per-query choice of *how much* to compress (variable budget, cost-aware) rather than which structure to use at a fixed budget.
- A measured answer to *when* per-query compression routing pays off on edge-class hardware (RQ2/RQ3).
- Do **not** claim that per-prompt routing between pruned variants is new.
- **The edge angle counts only if it is measured on edge hardware.** A Colab T4 is a datacenter GPU. Edge-cloud routing between *different* models already exists (ConsRoute, HybridFlow, Hybrid LLM). What is ours: compression-level routing evaluated under edge constraints, meaning
  - batch-1 inference;
  - a single resident model within a fixed memory budget (this is where the nested runtime, D5, matters);
  - energy per query;
  - router overhead on a weak CPU;
  - latency measured on at least one real edge-class device (target device: **[APPROVE]**).

## Formal setup
- Variants `V = {v0 (unpruned), v1, v2, v3}` ordered by cost.
- Outcome matrix `Y[q, v] ∈ {0,1}` (correct or not), plus soft score `S[q, v]` = normalised probability on the correct option (MCQ only). Cost `c(q, v)` = predicted latency (or energy) from a measured cost model.
- A router `π: q → v`. Accuracy `A(π) = mean_q Y[q, π(q)]`. Cost `C(π) = mean_q c(q, π(q)) + router overhead`.
- The learned router estimates `p̂_v(q) = P(Y[q, v] = 1 | q)` for every variant and decides `π_λ(q) = argmax_v  p̂_v(q) − λ · c̃(q, v)`. Sweeping `λ` traces a whole accuracy–cost **curve**, not one point.
- A router is only useful if its curve lies **above the static hull**: the line you get by randomly mixing two static variants (RouterBench's evaluation idea).

**Headline metrics**
1. Cost saving at iso-accuracy: the % reduction in latency/energy vs `v0` at accuracy ≥ `A(v0) − 1 pt`, and at ≥ 95% of `A(v0)`.
2. Area between the router curve and the static hull (normalised; RouterBench-style).
3. Headroom captured = `(A_router − A_hull) / (A_oracle − A_hull)` at matched cost.

This replaces v1's "≥ 90% of full-model accuracy" target. That target is trivially met by always using `v0`, so it can't show routing works.

## Decisions (v2)

### D1 — Base model: SmolLM2-1.7B (base) **[APPROVE: changes v1's Qwen2.5-1.5B default]**
These configs were checked from the Hugging Face `config.json` files:

| Model | Arch | Layers | Q / KV heads | head_dim | Biases | Embedding share | FLAP fit |
|---|---|---|---|---|---|---|---|
| **SmolLM2-1.7B** | `LlamaForCausalLM` | 24 | 32 / 32 (MHA) | 64 | none | ~6% (49k×2048, tied) | Native LLaMA layout, MHA → FLAP head pruning works as designed |
| Qwen2.5-1.5B | Qwen2 | 28 | 12 / **2** (GQA) | 128 | q/k/v bias | ~15% (152k×1536, tied) | Removing one KV head removes 6 query heads → attention can barely be pruned; mostly MLP-only FLAP |
| Llama-3.2-3B | Llama | 28 | 24 / 8 (GQA) | 128 | none | ~13% | Feasible with KV-group-aware pruning; 2× the memory and time |

- Why: SmolLM2-1.7B is the only edge-sized candidate with LLaMA naming **and** full multi-head attention, so FLAP's head pruning applies unmodified in spirit. Almost all its parameters (~94%) are prunable, and it is purpose-built for on-device use.
- Use the **base** model. FLAP's calibration (WikiText2) and lm-eval loglikelihood scoring are defined for base models, and published SmolLM2 base scores give a sanity check for `v0`. Not evaluating instruct-model behaviour is a stated limitation.
- Second model (Qwen2.5-1.5B or Llama-3.2-1B, both GQA) appears only as a P7 generalisation ablation, and only if the schedule allows.

### D2 — Port FLAP into `src/pruning/` instead of running the original repo as-is
Reading `CASIA-IVA-Lab/FLAP/lib/prune.py` and `main.py` shows it cannot run correctly on SmolLM2 (or Qwen) unmodified:
- `head_dim = 128` is hardcoded (`repeat_interleave(128)`, `reshape(..., 128)`). SmolLM2 uses 64.
- The AL-AM global ranking hardcodes `compression_weight = 512/3`. That is LLaMA-7B's parameters-per-head ÷ parameters-per-MLP-channel (`4·128/3`). For SmolLM2 it must be `4·head_dim/3 = 256/3`, otherwise the achieved pruning ratio is wrong.
- Bias compensation writes into `o_proj.bias` / `down_proj.bias`, which don't exist in bias-free models. Those layers must be swapped for bias-enabled `Linear`s.
- The repo pins `transformers==4.28` / `torch==1.10` (2023), which doesn't fit current Colab and may predate SmolLM2's config fields (e.g. `rope_theta`). Verify, but don't depend on it.
- `save_pretrained` of per-layer non-uniform shapes cannot be reloaded by `from_pretrained`, so we need our own save format (see D5).
- **Useful property:** importance scores are computed on the *unpruned* model's activations (`inps, outs = outs, inps # Use the original output`), and one global threshold sets the ratio. So for a fixed calibration set, the masks at different ratios are **nested**. D5 relies on this.

Port validation (all must pass before P2):
- (a) `0%` port produces logits identical to the base model.
- (b) Masked (`unstr`) and physically pruned models agree within fp16 tolerance.
- (c) The achieved parameter ratio equals the requested ratio within ±1%.
- (d) Masks are nested across ratios.
- (e) Reproduce the paper's *qualitative* claim: FLAP beats Wanda-SP and Magnitude-SP on WikiText2 PPL at 20% and 50%. Port those two methods too; they are about 30 lines each.

v1's "LLaMA-7B 50% PPL ≈ 31.8" check is dropped. A 7B model doesn't fit free-Colab RAM, and the number doesn't transfer to a 1.7B model.

### D3 — Pruning ratios chosen from a sweep, not fixed in advance **[APPROVE after P1 sweep]**
- Sweep FLAP at 10–60% in steps of 10. Each run is minutes. For each ratio, measure WikiText2 PPL, a quick lm-eval subset (~500 items/task), and batch-1 latency.
- Pick 4 variants (including 0%) such that:
  - each adjacent pair differs by ≥ ~10% measured latency;
  - the smallest variant is still above chance on most tasks.
- Default stays `0/20/35/50` unless the sweep argues otherwise.
- **Hardware-aligned pruning:** round each layer's kept MLP channels to a multiple of 64. Heads are atomic. This keeps tensor-core-friendly shapes. Measure speed with and without rounding (small ablation). Rounding preserves nesting.

### D4 — Query pool: run the source benchmarks through lm-evaluation-harness; use RouterBench as a secondary signal **[APPROVE]**
- RouterBench holds outputs from 11 *other* LLMs, none from ours, so every query has to be re-run on our variants anyway. lm-eval-harness gives standard prompts, standard scoring, per-query correctness via `--log_samples`, and published reference scores. RouterBench's own data are pickled 0-shot/5-shot frames meant for its `routerbench` package.
- **Tasks:**
  - RouterBench's scoreable core: MMLU, HellaSwag, WinoGrande, ARC-Challenge, GSM8K.
  - Easier tasks added: ARC-Easy, PIQA, BoolQ. Without easy queries there is little to route down.
  - Dropped: MBPP and MT-Bench, which need code execution or an LLM judge.
  - Size: roughly 35k items in total; cap MMLU and HellaSwag (e.g. 4k each) → ~15–18k queries.
- **Scoring:** MCQ by loglikelihood (`acc_norm` where that is the standard metric). GSM8K by greedy generation with lm-eval's default few-shot and a capped `max_gen_toks`. Store `Y`, `S` and output length per (query, variant).
- **RouterBench's role:** (i) it justifies the task choice and aligns the thesis with routing literature; (ii) its 11-LLM correctness gives an *external difficulty* label. Questions are matched by text (best effort). This lets P7 test the proposal's original idea: does difficulty measured on other LLMs transfer to pruned variants?
- **Splits:** for each task, a seeded 60/20/20 train/val/test split of query IDs, plus leave-one-task-out folds.
- The same outcome run **is** P1's zero-shot table, so one GPU job serves two phases.

### D5 — Deployment: nested shared-weight runtime ("one model, many depths")
v1 kept all 4 checkpoints resident in VRAM. That costs about Σ sizes ≈ 2.9× the full model, which *contradicts* the edge-memory motivation.

v2 primary design:
- Permute each layer's heads and MLP channels once by FLAP importance. The permutation leaves the function unchanged.
- Each variant is then a **prefix slice** of every layer, plus its own per-layer bias-compensation vectors and keep-counts (tens of KB per variant).
- Memory = 1× full model. Switching variant per query costs ~nothing.
- Store `o_proj`/`down_proj` transposed so the column slice is contiguous.
- Acceptance check: sliced-variant latency within ~5% of a physically pruned checkpoint.

Compare three deployment modes in P5:
- **A — Nested shared weights** (primary).
- **B — All variants resident.** Memory = Σ.
- **C — Single slot, swap on demand.** Memory = max, and the run pays the measured reload latency.

If A misses its speed target, fall back to B and report why. This addresses the memory flaw and is a concrete systems contribution.

### D6 — Router: per-variant success predictors with a cost knob
- **Features:**
  - Uses only the query's own text. Strip few-shot exemplars and templates, otherwise length just encodes task identity.
  - MiniLM-L6 sentence embedding (384-d).
  - Token length, digit/operator counts, MCQ flag and option count.
  - **The main router never sees the task ID.**
- **Models:** logistic regression (one per variant), a small multi-output MLP, and a kNN router (cosine, k≈20; a strong baseline in routing literature).
- **Calibration:** check reliability and Brier score; apply Platt/isotonic calibration on val if needed.
- **Decision rule:** `π_λ` from the formal setup. Choose operating points on **val** for each iso-accuracy target, then report on **test**.
- v1's "cheapest correct variant" 4-class label is kept as a secondary formulation, for the confusion matrix and an ablation. Its handling of unsolvable queries: no variant is correct → cheapest variant (the cost-optimal choice), documented.

### D7 — Baselines (all plotted on the same accuracy–cost axes)
1. Static `v0, v1, v2, v3`, plus the random-mixture **hull**.
2. **Task-oracle:** the best static variant per task, chosen on val. It needs the task ID. This is the bar that shows *content* routing works.
3. Rule-based: length and math/keyword heuristics with thresholds tuned on val.
4. Learned: LR, MLP, kNN.
5. **External-difficulty router:** trained on RouterBench's 11-LLM labels.
6. Random router.
7. **Per-query oracle** (upper bound).
8. (P6, optional) **Self-probe cascade:** run the smallest variant, compute the entropy of its answer-option distribution (MCQ) or of its first N tokens (GSM8K), escalate when uncertain. This is the RAP-style comparison point: RAP's trigger is memory pressure, this one's trigger is confidence.

### D8 — Measurement protocol
- **Accuracy:** from the outcome matrix. Decoding is greedy and deterministic, so every router is evaluated by **offline replay**, with no GPU needed per router or per λ. P4 confirms that replay matches live runs.
- **Latency cost model:**
  - Per variant, measure prefill latency over prompt lengths 64–2048 and decode ms/token, at batch 1, fp16, with CUDA sync, 10 warm-ups and the median of 30 runs.
  - Fit `cost(q, v) = prefill_v(len_q) + L_out(q, v) · decode_v + router_overhead`.
  - MCQ is deployed as "generate the answer letter" (prefill plus about 2 decode tokens). GSM8K uses its logged output length.
  - P4 validates the model on live runs; target error < 10% MAPE.
- **Energy:** the NVML cumulative energy counter, differenced around each run (verify it is exposed on Colab's T4). Fallback: power sampling every 20 ms. Report J/query with idle power subtracted and documented.
- **Memory:** model footprint per deployment mode plus `torch.cuda.max_memory_allocated` per query.
- **Edge realism (required — this carries the thesis's edge claim):**
  - Repeat the latency/energy profiling and the live end-to-end validation on at least one real edge-class device, e.g. a laptop CPU, a Raspberry Pi 5 (8 GB) or a Jetson Orin Nano.
  - Fit a separate cost model per device, and report router results under each device's cost model. The best routing policy may differ between a GPU and a CPU, which is a finding in itself.
  - Add an edge-budget scenario: a fixed memory cap (e.g. 4 GB) under which only deployment mode A fits all variants.
  - Energy on CPU via RAPL (Intel/AMD) or an external USB power meter.
- **Statistics:**
  - Paired bootstrap (10k resamples) on accuracy and cost differences.
  - McNemar's test for accuracy vs `v0`.
  - Router training over 5 seeds, reported as mean ± std.

### D9 — Framing within the resource-management literature
- The compression level is the **resource-allocation knob**. Per-query routing is a **resource-scheduling/provisioning** decision in Walia et al.'s taxonomy.
- The deployment modes (D5) are a **service-placement/provisioning** trade-off.
- MCDS (Tuli et al.) motivates the learned surrogate: our `p̂_v(q)` plays the role of their surrogate model, predicting the outcome of an allocation decision without executing it.

## Operating rules (unchanged in spirit)
- No GPU in the Claude session. All GPU work runs on Colab (T4, 16 GB) through `notebooks/run_colab.ipynb`, which only does `git pull` and calls scripts in `src/`. The user pastes logs and tracebacks back.
- One phase at a time. Don't start a phase until its inputs exist and are verified. Update `STATUS.md` after each phase.
- Every script:
  - is seeded and config-driven (`configs/*.yaml`);
  - writes to `results/<phase>/<run_id>/` (CSV/JSON plus a copy of `config.yaml` and the git hash);
  - stores checkpoints at the Drive paths given in the config;
  - finishes in under ~1.5 h and can resume.
- Pin the library stack in P0 (Colab's torch plus a recent `transformers`). Verify `LlamaAttention` infers the head count from tensor shape, so per-layer head counts work.
- Never invent numbers. Every metric comes from `results/`. Sanity checks: the `v0` lm-eval scores land near SmolLM2-1.7B's published base scores, and the port tests in D2 pass.
- Scope: the external-difficulty router and second-model ablations live in P7. The cascade (P6) and a custom edge dataset need user approval after P5.
- Ask the user before changing the base model, the labeling rule, the task set or the pruning ratios.

## Repo layout
```
src/qacr/{common,pruning,runtime,data,router,eval,profiling}/   library code (package "qacr")
scripts/            one entry-point script per experiment (pN_*.py), config-driven
configs/            YAML configs; ${ENV_VAR} / ${VAR:-default} expanded at load time
notebooks/run_colab.ipynb   only clones/pulls, installs, runs tests, runs a script, zips the run
results/<phase>/<run_id>/   imported run outputs (scripts/import_results.py <zip>)
tests/              pytest; must pass locally and on Colab before any GPU run
thesis/  notes/     writing and reference notes
STATUS.md  requirements.txt  pyproject.toml
```
- Local tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` (plugin autoload is disabled because the user's global site-packages has a broken `faker` pytest plugin).
- Every run writes `config.yaml`, `meta.json` (git commit + dirty flag), `env.json`, `pip_freeze.txt` and `run.log`. Results only count if the run was made from a clean commit.

## Known pitfalls (verified; do not reintroduce)
- **lm-eval 0.4.13 `samples=`:** results are mapped back by position, so indices **must be sorted**. `scripts/p2_outcomes.py` sorts them and the parser checks doc hashes.
- **GSM8K few-shot exemplars:** they depend on the few-shot seed **and** the evaluated doc list. Every variant must evaluate the identical sorted doc list with the same seed. `prompt_hash` is stored to verify this.
- **RouterBench:**
  - It is a pickle: load only via `qacr.data.routerbench.safe_load_pickle`, never `pd.read_pickle`.
  - It shuffles answer options, so matching keys sort options.
  - Its GSM8K items are from the train split, so they never match our test-split pool.
- **Truncation:** lm-eval left-truncates over-long contexts. Outcome runs flag `n_truncation_risk` per task.
- **Local environment:** use the repo-local `.venv` (git-ignored). A venv under %TEMP% was partly deleted by Windows temp cleanup mid-run (2026-10-21).
- **CPU timing:** only on mains power with the power state recorded per variant (`require_ac_power`); run each variant in a fresh process and repeat the first variant at the end as a drift check.
- **Run provenance:** git state is read before the run dir is created, and `results/` is excluded from the dirty check.

## Phases

| Phase | Weeks | GPU-h (est.) | Exit criterion |
|---|---|---|---|
| P0 Setup | 1 | 0.5 | Colab notebook runs `hello_gpu`, writes to Drive; NVML energy counter readable |
| P1 FLAP port + ladder | 1–4 | 6–8 | D2 tests pass; sweep table; FLAP vs Wanda-SP vs Mag-SP; ratios approved; latency/energy profiles (GPU + CPU) |
| P2 Query pool + outcome matrix | 4–7 | 6–8 | `data/{train,val,test}_ids`, `results/p2/outcomes.parquet [q, v, Y, S, out_len]`, **headroom report** |
| P3 Routers | 8–10 | 0.5 | All D7 routers callable as `route(q, λ) -> v`; val curves; calibration plots |
| P4 Runtime + pipeline | 11–14 | 3–5 | Nested runtime (D5-A) passes speed check; `Pipeline.run` logs everything; live vs replay agreement on ~1k test queries |
| P5 Evaluation | 15–18 | 1–2 | Curves, iso-accuracy savings, deployment-mode table, per-task breakdown, bootstrap CIs (use the `dataviz` skill) |
| P6 Cascade (optional) | 19–20 | 2–3 | Self-probe cascade on the same axes |
| P7 Ablations + thesis | 20–22 | 2–4 | Ablations below; every figure and table regenerated from `results/` by script |
| **Total** | | **~20–30** | |

**P2 decision gate:** if the per-query oracle beats the static hull by less than ~3 accuracy points at matched cost, stop and revisit the variant set (D3) with the user before building routers. That result on its own answers RQ2.

**P7 ablations:**
- Calibration data (WikiText2 vs C4).
- Embedding model (MiniLM vs bge-small).
- Hard vs soft labels (`Y` vs `S`).
- Variant-set size (3 / 4 / 6; cheap via replay if the sweep variants were evaluated).
- Hardware-aligned rounding on/off.
- External-difficulty labels.
- Leave-one-task-out generalisation.
- Second base model (if time allows).

## Risk handling
- **Colab disconnects:** resumable scripts, per-run checkpoints, runs under 1.5 h.
- **Nested runtime too slow:** fall back to deployment mode B and report the result.
- **Little headroom (P2 gate):** report it as a finding and adjust the variant spacing.
- **Learned router ≈ task-oracle:** a valid finding ("routing by task type is enough at this scale"). Frame RQ3 around it.
- **GPU quota:** the outcome matrix is computed once. All router experiments run on CPU via replay. Log GPU-hours in `STATUS.md`.

## Definition of done per phase
Code committed, tests passing, results files present, `STATUS.md` updated with metrics, GPU-hours and open issues, and the user told exactly what to run next on Colab.
