# Related work notes (P0)

Citation metadata below was verified on 2026-10-13:
- arXiv entries: titles, authors, dates and comments pulled from the arXiv API (`export.arxiv.org`).
- Journal papers: DOIs read from the PDFs in the thesis folder.
- Venue: stated only where a primary source confirms it. "Venue: unconfirmed" means: cite the arXiv version until checked.

## A. Framing: resource management in edge/fog/cloud

**[Walia24]** G. K. Walia, M. Kumar, S. S. Gill. *AI-Empowered Fog/Edge Resource Management for IoT Applications: A Comprehensive Review, Research Challenges, and Future Perspectives.* IEEE Communications Surveys & Tutorials, 26(1):619–, 2024. DOI 10.1109/COMST.2023.3338015.
- Survey of AI-based resource management, with 5 categories: provisioning, offloading, scheduling, placement, load balancing.
- **Use:** frames our router as a per-request *scheduling/provisioning* decision whose resource knob is the compression level. The deployment modes (shared weights / all resident / swap) are a *placement/provisioning* trade-off.
- Does not cover LLM inference workloads.

**[Tuli22]** S. Tuli, G. Casale, N. R. Jennings. *MCDS: AI Augmented Workflow Scheduling in Mobile Edge Cloud Computing Systems.* IEEE Transactions on Parallel and Distributed Systems, 33(11), Nov 2022. DOI 10.1109/TPDS.2021.3135907.
- Monte Carlo tree search plus a learned surrogate for non-myopic DAG scheduling.
- **Use:** precedent for a learned surrogate that predicts the outcome of an allocation decision without executing it. Our per-variant success predictor `p̂_v(q)` plays this role.

## B. Structured pruning (the compression side)

**[FLAP]** Y. An, X. Zhao, T. Yu, M. Tang, J. Wang. *Fluctuation-based Adaptive Structured Pruning for Large Language Models.* AAAI 2024. arXiv:2312.11983. Code: github.com/CASIA-IVA-Lab/FLAP.
- Retraining-free. Uses a fluctuation importance metric, standardised scores for adaptive per-layer ratios (AL-AM), and bias compensation.
- **Our base method.**
- Implementation facts that matter to us (read from `lib/prune.py`):
  - head_dim is hardcoded to 128;
  - the AL-AM weight is hardcoded to 512/3;
  - scores are computed on the unpruned model's activations, so masks are nested across ratios.

**[SliceGPT]** S. Ashkboos, M. L. Croci, M. Gennari do Nascimento, T. Hoefler, J. Hensman. *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024. arXiv:2401.15024.
- Alternative retraining-free structured method (rotation + deletion). Related work only.

**[Wanda]** M. Sun, Z. Liu, A. Bair, J. Z. Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024. arXiv:2306.11695.
- Weight × activation importance. Its structured variant (Wanda-SP, as implemented in the FLAP repo) is one of our P1 pruning baselines.

**[LLM-Pruner]** X. Ma, G. Fang, X. Wang. *LLM-Pruner: On the Structural Pruning of Large Language Models.* NeurIPS 2023. arXiv:2305.11627.
- Structured pruning with LoRA recovery. FLAP compares against it; we cite it for context only.

## C. Input-adaptive / elastic compression (closest prior work — must differentiate)

**[PuDDing]** J. Wee, M. Park, J. Lee. *Prompt-based Depth Pruning of Large Language Models.* ICML 2025 (ICML poster page). arXiv:2502.04348.
- A lightweight router picks one of several **depth**-pruned variants (which blocks to skip) per prompt, at a fixed sparsity.
- **Closest to us.**
- **Difference:** we choose *how much* to compress (variable budget, cost-aware), use width (head/channel) pruning via FLAP, share one weight copy across variants, and measure on edge hardware.

**[IFPruning]** B. Hou, Q. Chen, J. Wang, G. Yin, C. Wang, et al. *Instruction-Following Pruning for Large Language Models.* ICML 2025 (arXiv comment). arXiv:2501.02086.
- Per-prompt structured masks predicted by a trained predictor.
- **Difference:** requires training; we are training-free.

**[Flextron]** R. Cai, S. Muralidharan, G. Heinrich, H. Yin, Z. Wang, et al. *Flextron: Many-in-One Flexible Large Language Model.* ICML 2024 (project-page BibTeX). arXiv:2406.10260.
- Nested elastic LLM with input-adaptive routing.
- **Difference:** requires training/distillation of the elastic model. Our nesting comes for free from FLAP's global ranking.

**[MatFormer]** Devvrit, S. Kudugunta, A. Kusupati, T. Dettmers, K. Chen, et al. *MatFormer: Nested Transformer for Elastic Inference.* NeurIPS 2024 (arXiv comment). arXiv:2310.07707.
- Nested FFN sub-models trained jointly.
- **Use:** precedent for "one model, many sizes"; our nested runtime is a post-hoc, training-free analogue.

**[RAP]** H. Liu, C. Tian, X. Wei, Q. Li, L. Li. *RAP: Runtime Adaptive Pruning for LLM Inference.* arXiv:2505.17138, 2025. Venue: unconfirmed.
- An RL agent picks a pruning policy per request from sequence length, batch size and the available memory budget.
- **Difference:** its trigger is memory/workload, not query content or difficulty. Our P6 cascade is the closest comparison point.

## D. Query routing (the routing side)

**[RouterBench]** Q. J. Hu, J. Bieker, X. Li, N. Jiang, B. Keigwin, et al. *RouterBench: A Benchmark for Multi-LLM Routing System.* arXiv:2403.12031, 2024. Venue: unconfirmed (the proposal says ICML 2024; check before final citation).
- Benchmark of routing across 11 LLMs; introduces the convex-hull evaluation of routers.
- **Use:** the evaluation methodology (hull, AIQ-style area), task selection, and external difficulty labels for a P7 ablation.

**[RouteLLM]** I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, et al. *RouteLLM: Learning to Route LLMs with Preference Data.* arXiv:2406.18665, 2024. Venue: unconfirmed.
- Strong/weak model routing with a threshold knob.
- **Use:** precedent for cost-threshold curves.

**[HybridLLM]** D. Ding, A. Mallick, C. Wang, R. Sim, S. Mukherjee, et al. *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing.* ICLR 2024 (arXiv comment). arXiv:2404.14618.
- Difficulty-aware routing between a small and a large model.
- **Difference:** separate models, not compression levels of one model.

**[ConsRoute]** H. Qiao, H. Zhang, S. Mao, S. Cheng, J. Liu. *ConsRoute: Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device Large Language Models.* arXiv:2603.21237, 2026. Venue: unconfirmed.
- Routing across device/edge/cloud model tiers.
- **Difference:** separate models across tiers.

**[HybridFlow]** J. Dong, J. Li, T. Zheng, W. Lin. *HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference.* ICML 2026 (arXiv comment). arXiv:2512.22137.
- Subtask-level edge–cloud routing.
- **Difference:** separate edge/cloud models and subtask decomposition.

## E. Models, data and tools

**[SmolLM2]** L. Ben Allal, A. Lozhkov, E. Bakouch, G. Martín Blázquez, G. Penedo, et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model.* arXiv:2502.02737, 2025.
- Our base model, `HuggingFaceTB/SmolLM2-1.7B`.
- Config verified on the Hub: LlamaForCausalLM, 24 layers, hidden 2048, 32/32 heads, MLP 8192, no biases, tied embeddings, vocab 49152.

**[lm-eval]** EleutherAI *Language Model Evaluation Harness*, v0.4.13 (pinned in `requirements.txt`).
- Cite per the repository's CITATION file when writing.

## Positioning sentence (for the thesis introduction)

> Prior work routes prompts between *separate* models (RouterBench, RouteLLM, Hybrid LLM, ConsRoute, HybridFlow), selects *which* sub-structure to run at a fixed budget with training (PuDDing, IFPruning, Flextron), or adapts pruning to *memory pressure* (RAP). We study a training-free ladder of nested FLAP-pruned widths of one model and ask, under edge-device constraints, *when choosing the compression level per query pays off*.
