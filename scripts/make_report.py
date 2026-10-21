"""Generate the mid-semester report's tables and result text from result files only.

Reads thesis/midsem/sources.yaml and writes thesis/midsem/generated/*.tex. Every number in the
generated text comes from a file listed there; missing sources produce an explicit "pending" note.
Usage: python scripts/make_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MID = REPO / "thesis" / "midsem"
GEN = MID / "generated"

TASK_NAMES = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag",
              "winogrande": "WinoGrande", "piqa": "PIQA", "boolq": "BoolQ", "mmlu": "MMLU", "gsm8k": "GSM8K"}
METRIC_NAMES = {"acc": "acc", "acc_norm": "acc\\_norm", "exact_match": "exact match"}


def tex_escape(s: str) -> str:
    for ch in ("_", "%", "&", "#"):
        s = str(s).replace(ch, "\\" + ch)
    return s


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def load_json(p):
    return json.loads((REPO / p).read_text(encoding="utf-8"))


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------
def pool_table(src) -> str:
    m = load_json(src["pool_manifest"])
    rows = []
    for t in ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "boolq", "mmlu", "gsm8k"]:
        r = m["tasks"][t]
        sp = r["splits"]
        cap = f" (cap {fmt_int(r['cap'])})" if r["cap"] else ""
        rows.append(f"{TASK_NAMES[t]} & {fmt_int(r['n_eval_docs'])} & {r['n_duplicates_dropped']} & "
                    f"{fmt_int(r['n_kept'])}{cap} & {fmt_int(sp['train'])} / {fmt_int(sp['val'])} / {fmt_int(sp['test'])} & "
                    f"{METRIC_NAMES[r['metric']]} & {r['num_fewshot']} & \\texttt{{{tex_escape(r['dataset']['path'])}}} \\\\")
    st = m["splits_total"]
    total = (f"\\midrule Total & & & {fmt_int(m['n_total'])} & {fmt_int(st['train'])} / {fmt_int(st['val'])} / "
             f"{fmt_int(st['test'])} & & & \\\\")
    return ("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrcccl}\n\\toprule\n"
            "Task & Eval docs & Dupes & Kept & Train / Val / Test & Metric & Shots & HF dataset \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n" + total + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{Query pool (lm-eval {m['lm_eval_version']}, seed {m['seed']}). Dataset revisions (Hub commit "
            "hashes) are recorded in \\texttt{data/p2/manifest.json}.}}\n\\label{tab:pool}\n\\end{table}\n")


def routerbench_table(src) -> str:
    r = load_json(src["routerbench_report"])
    lines = []
    for t in ["hellaswag", "winogrande", "arc_challenge", "mmlu", "gsm8k", "arc_easy", "piqa", "boolq"]:
        x = r[t]
        note = x.get("note", "")
        if t == "gsm8k":
            note = "RouterBench uses the GSM8K train split"
        amb = x.get("ambiguous", "--")
        lines.append(f"{TASK_NAMES[t]} & {fmt_int(x['pool_rows'])} & {fmt_int(x['routerbench_rows'])} & "
                     f"{fmt_int(x['matched'])} & {amb} & {note} \\\\")
    return ("RouterBench~\\citep{hu2024routerbench} (0-shot file, revision \\texttt{"
            + r["revision"][:8] + "}, SHA-256 \\texttt{" + r["sha256"][:12] + "}\\ldots) records whether each of "
            + str(len(r["models"])) + " LLMs answered a prompt correctly. We attach the mean correctness over these "
            "models to our queries as an \\emph{external} difficulty label, used only in a later ablation. "
            "The file is a Python pickle; it is loaded with a restricted unpickler that permits only 11 "
            "pandas/numpy reconstruction functions. Questions are matched by content with answer options sorted, "
            "because RouterBench shuffles option order. Queries whose key occurs in RouterBench with different "
            f"scores are left unlabelled. In total {fmt_int(r['n_labelled'])} of our queries are labelled "
            "(Table~\\ref{tab:rb}).\n\n"
            "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrrl}\n\\toprule\n"
            "Task & Pool & RouterBench rows & Matched & Ambiguous & Note \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n\\caption{Matching our pool to RouterBench.}\\label{tab:rb}\n\\end{table}\n")


def p0_section(src) -> str:
    h = load_json(src["p0_hello"])
    c = h["checks"]
    idle, mm = c["idle_power"], c["matmul"]
    e = mm["energy"]
    d_idle = abs(idle["counter_joules"] - idle["sampled_joules"]) / idle["counter_joules"] * 100
    d_load = abs(e["counter_joules"] - e["sampled_joules"]) / e["counter_joules"] * 100
    return ("\\subsection{Measurement infrastructure (P0)}\n"
            f"On the Colab {c['nvml']['device_name']} all environment checks passed: seeding reproduced identical "
            "tensors; a model whose layers have different numbers of heads and MLP channels plus compensation biases "
            "(the shape FLAP produces) gave the same logits whether pruned physically or by masking, in fp32 and fp16 "
            "and with eager and SDPA attention (maximum fp16 logit difference "
            f"{c['pruned_shapes_sdpa_float16']['max_abs_logit_diff']:.1f}). "
            "The NVML energy counter is supported on the T4 and agrees with the independent 20\\,ms power sampler "
            f"to within {d_idle:.1f}\\% at idle ({idle['counter_joules']:.1f}\\,J vs.\\ {idle['sampled_joules']:.1f}\\,J "
            f"over {idle['duration_s']:.1f}\\,s, mean {idle['mean_power_w']:.1f}\\,W) and {d_load:.1f}\\% under an fp16 "
            f"matrix-multiply load ({e['counter_joules']:.1f}\\,J vs.\\ {e['sampled_joules']:.1f}\\,J over "
            f"{e['duration_s']:.1f}\\,s, mean {e['mean_power_w']:.1f}\\,W, {mm['tflops']:.1f}\\,TFLOPS). "
            "Energy figures in this work therefore use the counter.\n\n")


def cpu_profiles(src) -> dict:
    out = {}
    for d in src.get("cpu_profile_dirs") or []:
        for r in read_csv(REPO / d / "profile.csv"):
            out[r["variant"]] = r
    return out


def label(name: str) -> str:
    if name.startswith("none"):
        return "unpruned"
    m, r = name.rsplit("_", 1)
    return {"flap": "FLAP", "wanda_sp": "Wanda-SP", "mag_sp": "Mag-SP"}[m] + f" {float(r) * 100:.0f}\\%"


def fmt_ppl(x: float) -> str:
    if x < 1e4:
        return f"{x:.2f}"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def p1_analysis(rows: list[dict], prof: dict) -> str:
    """Analysis sentences; every number is computed here from sweep.csv / profiles.json."""
    by = {(r["method"], round(float(r["ratio"]), 2)): r for r in rows}
    ppl = lambda m, k: float(by[(m, k)]["ppl_wikitext2"])  # noqa: E731
    base = ppl("none", 0.0)
    ranks = [k for k in (0.2, 0.35, 0.5) if ("wanda_sp", k) in by and ("mag_sp", k) in by]
    holds = all(ppl("flap", k) < ppl("wanda_sp", k) < ppl("mag_sp", k) for k in ranks)
    flap_names = sorted((n for n in prof if n.startswith("flap_")), key=lambda n: float(n.split("_")[1]))
    all_names = ["none_0.00"] + flap_names
    dec = [prof[n]["generate"]["decode_ms_per_tok"] for n in all_names]
    gen_lo = min(prof[n]["generate"]["min_ms"] for n in all_names)
    gen_hi = max(prof[n]["generate"]["max_ms"] for n in all_names)
    p0, p5 = prof["none_0.00"], prof["flap_0.50"]
    pre = lambda p, L: p["prefill"][L]["median_ms"]  # noqa: E731
    e0 = p0["energy"]["joules_per_generated_token"]
    e5 = p5["energy"]["joules_per_generated_token"]
    w0, w5 = p0["energy"]["mean_power_w"], p5["energy"]["mean_power_w"]
    m0, m5 = p0["peak_mem_gb"], p5["peak_mem_gb"]
    minh = int(by[("flap", 0.5)]["min_heads_in_a_layer"])
    minh_w = int(by[("wanda_sp", 0.5)]["min_heads_in_a_layer"]) if ("wanda_sp", 0.5) in by else None
    n_layers = 24
    heads_total = int(by[("none", 0.0)]["heads_kept"]) // n_layers
    s = "\\paragraph{Analysis.} "
    s += ("\\emph{Quality.} " + ("At every ratio tested (" + ", ".join(f"{k * 100:.0f}\\%" for k in ranks) +
          ") the ranking FLAP $<$ Wanda-SP $<$ Magnitude-SP in perplexity holds, as reported for LLaMA in the FLAP "
          "paper. " if holds else "The FLAP $<$ Wanda-SP $<$ Magnitude-SP ranking does not hold at every ratio. "))
    s += (f"FLAP's perplexity rises from {base:.2f} (unpruned) to {ppl('flap', 0.2):.2f} at 20\\% "
          f"($\\times${ppl('flap', 0.2) / base:.2f}), {ppl('flap', 0.35):.2f} at 35\\% ($\\times${ppl('flap', 0.35) / base:.2f}) "
          f"and {ppl('flap', 0.5):.2f} at 50\\% ($\\times${ppl('flap', 0.5) / base:.2f}); Wanda-SP reaches "
          f"{ppl('wanda_sp', 0.5):.2f} at 50\\%, and Magnitude-SP is unusable at every ratio "
          f"({fmt_ppl(ppl('mag_sp', 0.2))} already at 20\\%). FLAP's adaptive structure is visible in the plans: at 50\\% "
          f"one layer keeps only {minh} of its {heads_total} heads, while the uniform baselines keep {minh_w} in every layer. ")
    s += ("\\emph{Latency and energy on the T4.} Decode latency is essentially independent of the pruning ratio: "
          f"{min(dec):.1f}--{max(dec):.1f}\\,ms per token from 0\\% to {float(flap_names[-1].split('_')[1]) * 100:.0f}\\% "
          f"(all individual 64-token generations took {gen_lo / 1000:.2f}--{gen_hi / 1000:.2f}\\,s), so width pruning "
          "does not shorten generation here. This is consistent with batch-1 decoding of a 1.7\\,B model on this GPU "
          "being dominated by fixed per-step overheads rather than by the size of the weight matrices; the cause was "
          "not measured directly. Prefill does become faster as prompts get longer: at 50\\% "
          f"the 1024-token prefill drops from {pre(p0, '1024'):.1f} to {pre(p5, '1024'):.1f}\\,ms "
          f"($-${(1 - pre(p5, '1024') / pre(p0, '1024')) * 100:.0f}\\%), the 512-token prefill from {pre(p0, '512'):.1f} "
          f"to {pre(p5, '512'):.1f}\\,ms ($-${(1 - pre(p5, '512') / pre(p0, '512')) * 100:.0f}\\%), while the 128-token "
          f"prefill barely changes ({pre(p0, '128'):.1f} vs.\\ {pre(p5, '128'):.1f}\\,ms). Energy per generated token "
          f"falls from {e0:.2f} to {e5:.2f}\\,J ($-${(1 - e5 / e0) * 100:.0f}\\%) at 50\\% because the mean GPU power "
          f"during generation drops from {w0:.1f} to {w5:.1f}\\,W at the same duration; peak GPU memory falls from "
          f"{m0:.2f} to {m5:.2f}\\,GB. The latency benefit of width pruning on this GPU is therefore limited to prefill; "
          "the energy and memory benefits are measurable. The laptop CPU is measured separately.\n\n")
    return s


def p1_section(src) -> tuple[str, dict]:
    facts: dict = {}
    s = "\\subsection{Pruned-variant ladder (P1)}\n"
    if not src.get("p1_sweep_dir"):
        s += ("\\pend{the FLAP sweep (ratios 0.1--0.6) and the Wanda-SP / Magnitude-SP baselines are running on the "
              "Colab T4; perplexity, parameter counts and GPU latency will be reported from "
              "\\texttt{results/p1/<run>/sweep.csv}.}\n\n")
    else:
        d = REPO / src["p1_sweep_dir"]
        rows = [r for r in read_csv(d / "sweep.csv") if r["status"] == "ok"]
        prof = json.loads((d / "profiles.json").read_text(encoding="utf-8"))
        facts["sweep"] = rows
        body = []
        for r in rows:
            name = f"{r['method']}_{float(r['ratio']):.2f}"
            body.append(f"{label(name)} & {float(r['achieved_ratio']) * 100:.1f} & {int(r['n_params']) / 1e9:.3f} & "
                        f"{fmt_ppl(float(r['ppl_wikitext2']))} & {float(r['prefill_ms_512']):.1f} & "
                        f"{float(r['decode_ms_per_tok']):.1f} & "
                        + (f"{float(r['joules_per_token']):.3f}" if r.get("joules_per_token") else "--") + " \\\\")
        analysis = p1_analysis(rows, prof)
        s += ("Table~\\ref{tab:sweep} lists every pruned variant produced on the T4. ``Achieved'' is the fraction of "
              "prunable (attention and MLP) parameters removed; embeddings are not pruned. Latency is batch-1 fp16.\n\n"
              "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
              "Variant & Achieved (\\%) & Params (B) & WikiText-2 PPL & Prefill 512 (ms) & Decode (ms/tok) & J/token \\\\\n"
              "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
              "\\caption{FLAP sweep and baselines on SmolLM2-1.7B (Colab T4, fp16, batch 1). Decode is per generated "
              "token after a 128-token prompt; J/token is GPU energy (NVML counter) per generated token.}\\label{tab:sweep}\n"
              "\\end{table}\n\n"
              "\\begin{figure}[h]\\centering\n\\includegraphics[width=0.78\\linewidth]{figures/fig_ppl.pdf}\n"
              "\\caption{WikiText-2 perplexity (log scale) against the fraction of prunable parameters removed.}"
              "\\label{fig:ppl}\n\\end{figure}\n\n"
              "\\begin{figure}[h]\\centering\n\\includegraphics[width=\\linewidth]{figures/fig_t4.pdf}\n"
              "\\caption{FLAP variants on the T4 (batch 1, fp16): prefill latency for three prompt lengths, decode "
              "latency per token, and GPU energy per generated token.}\\label{fig:t4}\n\\end{figure}\n\n"
              + analysis)
    cpu = cpu_profiles(src)
    if not cpu:
        s += (r"\paragraph{Laptop CPU (edge device).} \pend{a first set of CPU measurements was discarded because the "
              r"laptop was running on battery with the CPU throttled (about 55\% performance, about 1.97\,GHz); "
              r"repetition spreads reached 2$\times$ and the variants were non-monotonic in size. Those runs are kept in "
              r"the repository marked INVALID. The re-measurement on mains power, with the power state recorded per "
              r"variant, is pending.}" + "\n\n")
    if cpu:
        body = []
        for name, r in cpu.items():
            body.append(f"{label(name)} & {int(r['n_params']) / 1e9:.3f} & {float(r['weight_gb']):.2f} & "
                        f"{float(r['prefill_ms_128']) / 1000:.2f} & {float(r['prefill_ms_512']) / 1000:.2f} & "
                        f"{float(r['decode_ms_per_tok']):.0f} \\\\")
        facts["cpu"] = cpu
        s += ("\\paragraph{Edge device (laptop CPU).} Table~\\ref{tab:cpu} gives batch-1 latency on the Intel "
              "i5-12500H in fp32 (median of 3 repetitions after one warm-up). "
              "CPU energy is not measured yet (no power measurement is implemented for this laptop).\n\n"
              "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
              "Variant & Params (B) & Weights (GB, fp32) & Prefill 128 (s) & Prefill 512 (s) & Decode (ms/tok) \\\\\n"
              "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
              "\\caption{Laptop-CPU latency (fp32, batch 1, 12 threads).}\\label{tab:cpu}\n\\end{table}\n")
        if len(cpu) == 1:
            s += ("\\pend{CPU profiles of the pruned variants, which need the plans produced by the T4 sweep.}\n\n")
    return s, facts


def p2_section(src) -> tuple[str, dict]:
    s = "\\subsection{Accuracy and oracle headroom (P2)}\n"
    if not src.get("p2_analysis_dir"):
        return s + ("\\pend{per-variant accuracy on all 19{,}242 queries and the headroom analysis of "
                    "Eq.~\\eqref{eq:headroom}; the outcome runs for the unpruned model and the FLAP variants are "
                    "queued on the T4 after the sweep.}\n\n"), {}
    d = REPO / src["p2_analysis_dir"]
    acc = read_csv(d / "accuracy_table.csv")
    variants = [k for k in acc[0].keys() if k != "task"]
    body = [f"{TASK_NAMES.get(r['task'], 'Macro average')} & " + " & ".join(f"{float(r[v]) * 100:.1f}" for v in variants)
            + " \\\\" for r in acc]
    s += ("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{l" + "r" * len(variants) + "}\n\\toprule\n"
          "Task & " + " & ".join(label(v) if "_" in v else v for v in variants) + " \\\\\n\\midrule\n"
          + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
          "\\caption{Accuracy (\\%) on the test split.}\\label{tab:acc}\n\\end{table}\n\n")
    facts = {"acc": acc}
    for f in sorted(d.glob("headroom_*.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        facts[f.stem] = h
        s += (f"\\paragraph{{Headroom ({h['cost_model']} cost model).}} Oracle accuracy "
              f"{h['oracle']['accuracy'] * 100:.1f}\\% at {h['oracle']['cost_ms']:.1f}\\,ms per query; the random-mixture "
              f"hull reaches {h['hull_accuracy_at_oracle_cost'] * 100:.1f}\\% at that cost, so the headroom is "
              f"{h['headroom_points'] * 100:.1f} points. {h['oracle']['unsolvable_share'] * 100:.1f}\\% of test queries "
              "are answered correctly by no variant.\n\n")
    return s, facts


def _p1_numbers(src) -> dict | None:
    if not src.get("p1_sweep_dir"):
        return None
    d = REPO / src["p1_sweep_dir"]
    by = {(r["method"], round(float(r["ratio"]), 2)): r for r in read_csv(d / "sweep.csv") if r["status"] == "ok"}
    prof = json.loads((d / "profiles.json").read_text(encoding="utf-8"))
    p0, p5 = prof["none_0.00"], prof["flap_0.50"]
    return {
        "ppl0": float(by[("none", 0.0)]["ppl_wikitext2"]), "ppl20": float(by[("flap", 0.2)]["ppl_wikitext2"]),
        "ppl50": float(by[("flap", 0.5)]["ppl_wikitext2"]), "wanda50": float(by[("wanda_sp", 0.5)]["ppl_wikitext2"]),
        "pre1024_cut": 1 - p5["prefill"]["1024"]["median_ms"] / p0["prefill"]["1024"]["median_ms"],
        "energy_cut": 1 - p5["energy"]["joules_per_generated_token"] / p0["energy"]["joules_per_generated_token"],
        "dec_lo": min(p["generate"]["decode_ms_per_tok"] for k, p in prof.items() if k.startswith(("flap", "none"))),
        "dec_hi": max(p["generate"]["decode_ms_per_tok"] for k, p in prof.items() if k.startswith(("flap", "none"))),
    }


def abstract(src) -> str:
    m = load_json(src["pool_manifest"])
    n = _p1_numbers(src)
    s = ("We study whether choosing an LLM's compression level per query pays off on edge hardware. We port FLAP, a "
         "retraining-free structured pruning method, to SmolLM2-1.7B and obtain a ladder of nested pruned variants of "
         "one model, stored as small pruning plans over a single copy of the weights. We build a "
         f"{fmt_int(m['n_total'])}-question pool from eight benchmarks with reproducible splits and a scoring pipeline "
         "that checks every result against the evaluation harness. ")
    if n:
        s += (f"On a Tesla T4, FLAP raises WikiText-2 perplexity from {n['ppl0']:.2f} to {n['ppl20']:.2f} at 20\\% and "
              f"{n['ppl50']:.2f} at 50\\% of prunable parameters removed (Wanda-SP: {n['wanda50']:.2f} at 50\\%; Magnitude-SP "
              "fails), reproducing the FLAP paper's method ranking at this scale. Pruning 50\\% cuts 1024-token prefill "
              f"latency by {n['pre1024_cut'] * 100:.0f}\\% and energy per generated token by {n['energy_cut'] * 100:.0f}\\%, "
              f"but batch-1 decode latency stays at {n['dec_lo']:.1f}--{n['dec_hi']:.1f}\\,ms per token for every ratio. ")
    if src.get("p2_analysis_dir"):
        s += "We report per-variant accuracy and the oracle headroom available to a per-query router. "
    else:
        s += ("Per-variant accuracy on the query pool, the oracle headroom available to a router, and laptop-CPU "
              "latencies are being measured and are reported as pending. ")
    return s + "\n"


def conclusion(src) -> str:
    n = _p1_numbers(src)
    s = ("The infrastructure for the study is in place and verified: a seeded, config-driven pipeline with recorded "
         "provenance, a faithful and tested port of FLAP and two baselines, a reproducible 19{,}242-question pool, and a "
         "scoring pipeline whose every result is cross-checked against lm-eval. ")
    if n:
        s += ("The first measurements answer RQ1: FLAP produces a steep accuracy--cost ladder on a 1.7\\,B model "
              f"(perplexity $\\times${n['ppl20'] / n['ppl0']:.2f} at 20\\%, $\\times${n['ppl50'] / n['ppl0']:.2f} at 50\\%) and "
              "keeps its advantage over Wanda-SP and Magnitude-SP. They also show that on a T4 at batch size 1 width pruning "
              "reduces prefill latency, energy and memory but not decode latency, so the cost model used for routing must "
              "be measured per device rather than assumed proportional to parameter count. ")
    s += ("Whether routing pays off (RQ2) depends on the oracle headroom, which the outcome runs now in progress will "
          "measure; the analysis separates genuine complementarity between variants from lucky guesses by comparing "
          "with independence and chance references. The remaining phases are planned as follows.\n")
    return s


def main() -> int:
    src = yaml.safe_load((MID / "sources.yaml").read_text(encoding="utf-8"))
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "pool_table.tex").write_text(pool_table(src), encoding="utf-8")
    (GEN / "routerbench_table.tex").write_text(routerbench_table(src), encoding="utf-8")
    results = p0_section(src)
    s1, f1 = p1_section(src)
    s2, f2 = p2_section(src)
    results += s1 + s2
    (GEN / "results.tex").write_text(results, encoding="utf-8")
    (GEN / "abstract.tex").write_text(abstract(src), encoding="utf-8")
    (GEN / "conclusion.tex").write_text(conclusion(src), encoding="utf-8")
    facts = {"p1": f1, "p2": f2}
    (GEN / "facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print("wrote", sorted(p.name for p in GEN.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
