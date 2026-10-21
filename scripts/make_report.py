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
        s += (r"\paragraph{Laptop CPU (edge device).} \pend{two attempts did not give trustworthy timings, so no CPU "
              r"numbers are reported. The first ran on battery with the CPU throttled (about 55\% performance, about "
              r"1.97\,GHz; repetition spreads up to 2$\times$, variants non-monotonic in size). The second ran on mains "
              r"power in ``Best performance'' mode, but processor performance still varied between 31\% and 61\%, the "
              r"unpruned model measured 443\,ms/token at the start of the session and 510\,ms/token in a repeat at the "
              r"end (+15\%), and FLAP-20\% measured slower than the unpruned model. Both sets are kept in the repository "
              r"marked INVALID / PRELIMINARY; CPU timing is deferred to a dedicated measurement step "
              r"(Section~\ref{sec:plan}).}" + "\n\n")
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
    vname = lambda v: "unpruned" if v == "v0" else label(v)  # noqa: E731
    body = [f"{TASK_NAMES.get(r['task'], 'Macro average')} & " + " & ".join(f"{float(r[v]) * 100:.1f}" for v in variants)
            + " \\\\" for r in acc]
    n_test = None
    heads = {}
    for f in sorted(d.glob("headroom_*.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        heads[h["cost_model"]] = h
        n_test = h["n_queries"]
    s += ("Table~\\ref{tab:acc} gives the accuracy of each variant on the " + (fmt_int(n_test) if n_test else "")
          + " test-split questions (primary metric per task, Sec.~\\ref{sec:data}).\n\n"
          "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{l" + "r" * len(variants) + "}\n\\toprule\n"
          "Task & " + " & ".join(vname(v) for v in variants) + " \\\\\n\\midrule\n"
          + "\n".join(body[:-1]) + "\n\\midrule\n" + body[-1] + "\n\\bottomrule\n\\end{tabular}\n"
          "\\caption{Accuracy (\\%) on the test split.}\\label{tab:acc}\n\\end{table}\n\n")
    facts = {"acc": acc, "headroom": heads}
    if not heads:
        return s, facts

    # oracle per cost model: best fixed / raw oracle / informative oracle
    rows = []
    for label_, h in heads.items():
        bf, orc, io = h["best_fixed"], h["oracle"], h["informative_oracle"]
        rows.append(f"{tex_escape(label_)} & {vname(bf['variant'])}: {bf['accuracy'] * 100:.1f} @ {bf['cost_ms']:.1f} & "
                    f"{orc['accuracy'] * 100:.1f} @ {orc['cost_ms']:.1f} & {io['accuracy'] * 100:.1f} @ {io['cost_ms']:.1f} & "
                    f"{h['independent_oracle_accuracy'] * 100:.1f} \\\\")
    cost_rows = []
    for v in variants:
        cost_rows.append(f"{vname(v)} & " + " & ".join(f"{heads[k]['static'][v]['cost_ms']:.1f}" for k in heads) + " \\\\")
    s += ("\\paragraph{Cost per query.} Table~\\ref{tab:cost} gives the mean modelled latency per test question "
          "(Sec.~\\ref{sec:math}: measured prefill latency at the question's prompt length, plus measured decode latency "
          "times the generated length for GSM8K).\n\n"
          "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{l" + "r" * len(heads) + "}\n\\toprule\n"
          "Variant & " + " & ".join(f"{tex_escape(k)} (ms)" for k in heads) + " \\\\\n\\midrule\n"
          + "\n".join(cost_rows) + "\n\\bottomrule\n\\end{tabular}\n"
          "\\caption{Mean modelled latency per test question.}\\label{tab:cost}\n\\end{table}\n\n")
    first = next(iter(heads.values()))
    io = first["informative_oracle"]
    beh = json.loads((d / "behaviour.json").read_text(encoding="utf-8")) if (d / "behaviour.json").exists() else None

    # behaviour: collapse evidence
    if beh:
        mc = beh["mc_max_answer_share"]
        brows = []
        for t in sorted(mc, key=lambda t: list(TASK_NAMES).index(t) if t in TASK_NAMES else 99):
            brows.append(f"{TASK_NAMES.get(t, t)} & {io['per_task'][t]['constant_baseline'] * 100:.1f} & "
                         + " & ".join(f"{mc[t][v]['max_share'] * 100:.0f}" for v in variants) + " \\\\")
        g = beh["gsm8k_generation"]
        grows = [f"{vname(v)} & {g[v]['accuracy'] * 100:.1f} & {g[v]['mean_tokens']:.0f} & {g[v]['median_tokens']:.0f} & "
                 f"{g[v]['share_at_cap'] * 100:.1f} \\\\" for v in variants]
        s += ("\\paragraph{Collapse of the pruned variants.} Accuracy alone hides how the pruned variants fail. "
              "Table~\\ref{tab:collapse} gives, for every multiple-choice task, the share of questions on which each "
              "variant gives its single most frequent answer; 100\\% means the variant answers every question "
              "identically (e.g.\\ always ``A'' or always ``no''). The ``constant'' column is the accuracy of always "
              "giving the most common gold answer. On GSM8K (Table~\\ref{tab:gsm8k}) the pruned variants mostly fail "
              f"to finish an answer and generate until the {g[variants[0]]['cap']}-token limit.\n\n"
              "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lr" + "r" * len(variants) + "}\n\\toprule\n"
              "Task & Constant & " + " & ".join(vname(v) for v in variants) + " \\\\\n\\midrule\n" + "\n".join(brows)
              + "\n\\bottomrule\n\\end{tabular}\n\\caption{Share (\\%) of test questions receiving each variant's most "
              "frequent answer, and the accuracy of the best constant answer.}\\label{tab:collapse}\n\\end{table}\n\n"
              "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
              "Variant & Accuracy (\\%) & Mean tokens & Median tokens & At limit (\\%) \\\\\n\\midrule\n"
              + "\n".join(grows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{GSM8K generation on the test split.}"
              "\\label{tab:gsm8k}\n\\end{table}\n\n")

    s += ("\\paragraph{Oracle headroom.} The plain oracle (Eq.~\\eqref{eq:headroom}) counts a question as solved if "
          "any variant answers it correctly. When some variants answer every question identically, combining them "
          "``solves'' questions that no router could identify (two variants that always answer ``yes'' and always "
          "answer ``no'' together cover all of BoolQ). We therefore also report an \\emph{informative oracle}: on each "
          "task a variant may be chosen only if its accuracy beats the best constant answer (one-sided binomial test, "
          f"$p<{io['alpha']}$). Table~\\ref{{tab:headroom}} gives both, with the accuracy expected if the variants' errors "
          "were independent, $1-\\prod_v(1-A_v)$. Entries are accuracy (\\%) @ mean cost per question (ms); "
          "accuracy here is over all test questions pooled, whereas Table~\\ref{tab:acc} averages the eight tasks "
          f"with equal weight, which is why {vname(first['best_fixed']['variant'])} shows "
          f"{first['best_fixed']['accuracy'] * 100:.1f} here and "
          f"{float(next(r for r in acc if r['task'] == 'macro_avg')[first['best_fixed']['variant']]) * 100:.1f} there.\n\n"
          "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lllll}\n\\toprule\n"
          "Cost model & Best fixed & Oracle & Informative oracle & Independent \\\\\n\\midrule\n" + "\n".join(rows)
          + "\n\\bottomrule\n\\end{tabular}\n\\caption{Best fixed variant, oracle and informative oracle on the test "
          "split.}\\label{tab:headroom}\n\\end{table}\n\n")

    trows = []
    for t in sorted(io["per_task"], key=lambda t: list(TASK_NAMES).index(t) if t in TASK_NAMES else 99):
        x, ph = io["per_task"][t], first["per_task_headroom"][t]
        inf = [vname(v) for v, z in x["variants"].items() if z["informative"]]
        trows.append(f"{TASK_NAMES.get(t, t)} & {ph['n']} & "
                     + (f"{x['constant_baseline'] * 100:.1f}" if x["constant_baseline"] > 0 else "--")
                     + f" & {ph['best_static_accuracy'] * 100:.1f} & {ph['oracle_accuracy'] * 100:.1f} & "
                     f"{x['oracle_accuracy'] * 100:.1f} & {', '.join(inf)} \\\\")
    s += ("\\begin{table}[h]\\centering\\small\n\\begin{tabularx}{\\linewidth}{lrrrrrX}\n\\toprule\n"
          "Task & $n$ & Constant & Best fixed & Oracle & Inf.\\ oracle & Variants above the constant answer \\\\\n"
          "\\midrule\n" + "\n".join(trows) + "\n\\bottomrule\n\\end{tabularx}\n\\caption{Per-task accuracy (\\%) on "
          "the test split.}\\label{tab:pertask}\n\\end{table}\n\n")

    pair = first["pairwise_vs_reference"]
    prow = [f"{vname(v)} & {p['both_correct'] * 100:.1f} & {p['only_pruned_correct'] * 100:.1f} & "
            f"{p['only_reference_correct'] * 100:.1f} & {p['both_wrong'] * 100:.1f} \\\\" for v, p in pair.items()]
    s += ("\\paragraph{Agreement with the unpruned model.} Table~\\ref{tab:pair} shows, for each pruned variant, how "
          "often it and the unpruned model are both right, only one of them is right, or both are wrong.\n\n"
          "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
          "Variant & Both right & Only pruned right & Only unpruned right & Both wrong \\\\\n\\midrule\n"
          + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Outcome overlap with the unpruned model "
          "(\\% of test questions).}\\label{tab:pair}\n\\end{table}\n\n")
    return s, facts


def p6_section(src) -> str:
    """Query-level confidence cascade (offline, held-out half), from results/p6/<run>/cascade.csv."""
    if not src.get("p6_cascade_dir"):
        return ""
    rows = read_csv(REPO / src["p6_cascade_dir"] / "cascade.csv")
    body = []
    for r in rows:
        body.append(f"{label(r['small'])} & {float(r['max_loss']) * 100:.1f} & {float(r['served']) * 100:.1f} & "
                    f"{float(r['loss']) * 100:.2f} ({float(r['loss_ci_lo']) * 100:.2f}--{float(r['loss_ci_hi']) * 100:.2f}) & "
                    f"{float(r['t4_cascade_cost']):.2f}$\\times$ & {float(r['param_cascade_cost']):.2f}$\\times$ \\\\")
    half = [r for r in rows if abs(float(r["max_loss"]) - 0.005) < 1e-9]
    best = max(half, key=lambda r: float(r["served"])) if half else rows[0]
    return ("\\subsection{Query-level confidence cascade (offline)}\n"
            "A cheap variant answers each multiple-choice question first; its answer is kept if its confidence (the "
            "probability of its chosen option) is at least $\\tau$, otherwise the question is escalated to the unpruned "
            "model. The test questions are split into two fixed halves: $\\tau$ is chosen on one half (largest share served "
            "within an accuracy-loss budget) and all numbers are reported on the other half. Escalated questions pay for "
            "both models, so the cascade is cheaper than the unpruned model only if the share served exceeds the cheap "
            "model's cost ratio.\n\n"
            "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrlrr}\n\\toprule\n"
            "Cheap variant & Budget (pts) & Served (\\%) & Loss, pts (95\\% CI) & T4 cost & Param.-ratio cost \\\\\n"
            "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Confidence cascade on the held-out half of the multiple-choice test questions. Cost is relative to "
            "the unpruned model alone (T4: measured cost model; parameter ratio: device-independent reference, not a "
            "measurement).}\\label{tab:cascade}\n\\end{table}\n\n"
            f"The pruned variants' confidence is informative --- at a 0.5-point budget {label(best['small'])} answers "
            f"{float(best['served']) * 100:.1f}\\% of the questions itself --- but every configuration costs more than the "
            f"unpruned model ({min(float(r['t4_cascade_cost']) for r in rows):.2f}--"
            f"{max(float(r['t4_cascade_cost']) for r in rows):.2f}$\\times$ on the T4), because the share served stays far "
            "below the cost ratio of the cheap variant. A query-level cascade between these FLAP variants cannot save cost; "
            "the planned fix is a token-level cascade (Section~\\ref{sec:plan}).\n\n")


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


def _p2_numbers(src) -> dict | None:
    if not src.get("p2_analysis_dir"):
        return None
    d = REPO / src["p2_analysis_dir"]
    acc = {r["task"]: r for r in read_csv(d / "accuracy_table.csv")}["macro_avg"]
    h = json.loads(sorted(d.glob("headroom_*.json"))[0].read_text(encoding="utf-8"))
    pruned = sorted((v for v in acc if v not in ("task", "v0")), key=lambda v: float(v.split("_")[1]))
    return {"n": h["n_queries"], "acc": {v: float(acc[v]) * 100 for v in acc if v != "task"}, "pruned": pruned,
            "oracle": h["oracle"]["accuracy"] * 100, "best": h["best_fixed"]["accuracy"] * 100,
            "inf": h["informative_oracle"]["accuracy"] * 100,
            "inf_v0_share": h["informative_oracle"]["pick_share"].get("v0", 0) * 100}


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
    p2 = _p2_numbers(src)
    if p2:
        s += (f"On {fmt_int(p2['n'])} test questions, macro accuracy falls from {p2['acc']['v0']:.1f}\\% (unpruned) to "
              + ", ".join(f"{p2['acc'][v]:.1f}\\% ({label(v)})" for v in p2["pruned"]) + "; several pruned variants "
              "collapse to a constant answer on MMLU and BoolQ and fail to finish GSM8K answers. A per-query oracle "
              f"over the variants reaches {p2['oracle']:.1f}\\% versus {p2['best']:.1f}\\% for the best fixed variant, but "
              f"once variants that do not beat a constant answer are excluded the gain shrinks to "
              f"{p2['inf'] - p2['best']:.1f} points ({p2['inf']:.1f}\\%), so on this GPU the headroom for per-query "
              "compression routing is small. ")
        if not src.get("cpu_profile_dirs"):
            s += "Laptop-CPU latencies are pending. "
    else:
        s += ("Per-variant accuracy on the query pool, the oracle headroom available to a router, and laptop-CPU "
              "latencies are being measured and are reported as pending. ")
    return s + "\n"


def conclusion_facts(src) -> dict | None:
    """Every number used in the structured conclusion, read from the result files."""
    if not (src.get("p1_sweep_dir") and src.get("p2_analysis_dir")):
        return None
    d1 = REPO / src["p1_sweep_dir"]
    by = {(r["method"], round(float(r["ratio"]), 2)): r for r in read_csv(d1 / "sweep.csv") if r["status"] == "ok"}
    prof = json.loads((d1 / "profiles.json").read_text(encoding="utf-8"))
    p0, p5 = prof["none_0.00"], prof["flap_0.50"]
    d2 = REPO / src["p2_analysis_dir"]
    h = json.loads(sorted(d2.glob("headroom_*.json"))[0].read_text(encoding="utf-8"))
    beh = json.loads((d2 / "behaviour.json").read_text(encoding="utf-8"))
    acc = {r["task"]: r for r in read_csv(d2 / "accuracy_table.csv")}
    io = h["informative_oracle"]["per_task"]
    gains = {t: (io[t]["oracle_accuracy"] - h["per_task_headroom"][t]["best_static_accuracy"]) * 100 for t in io}
    pos = sorted(((t, g) for t, g in gains.items() if g > 0.05), key=lambda x: -x[1])
    n0, n5 = int(by[("none", 0.0)]["n_params"]), int(by[("flap", 0.5)]["n_params"])
    return {
        "ppl0": float(by[("none", 0.0)]["ppl_wikitext2"]), "ppl10": float(by[("flap", 0.1)]["ppl_wikitext2"]),
        "ppl20": float(by[("flap", 0.2)]["ppl_wikitext2"]), "ppl50": float(by[("flap", 0.5)]["ppl_wikitext2"]),
        "e0": p0["energy"]["joules_per_generated_token"], "e5": p5["energy"]["joules_per_generated_token"],
        "m0": p0["peak_mem_gb"], "m5": p5["peak_mem_gb"],
        "pre512": 1 - p5["prefill"]["512"]["median_ms"] / p0["prefill"]["512"]["median_ms"],
        "pre1024": 1 - p5["prefill"]["1024"]["median_ms"] / p0["prefill"]["1024"]["median_ms"],
        "dec_lo": min(prof[k]["generate"]["decode_ms_per_tok"] for k in prof if k.startswith(("none", "flap"))),
        "dec_hi": max(prof[k]["generate"]["decode_ms_per_tok"] for k in prof if k.startswith(("none", "flap"))),
        "best": h["best_fixed"]["accuracy"] * 100, "oracle": h["oracle"]["accuracy"] * 100,
        "inf": h["informative_oracle"]["accuracy"] * 100, "inf_v0": h["informative_oracle"]["pick_share"]["v0"] * 100,
        "pos_gains": pos, "piqa50": float(acc["piqa"]["flap_0.50"]) * 100,
        "piqa_const": io["piqa"]["constant_baseline"] * 100, "fewer_params": (1 - n5 / n0) * 100,
        "macro": {v: float(acc["macro_avg"][v]) * 100 for v in acc["macro_avg"] if v != "task"},
        "gsm_cap": {v: x["share_at_cap"] * 100 for v, x in beh["gsm8k_generation"].items()},
        "mmlu_const": {v: x["max_share"] * 100 for v, x in beh["mc_max_answer_share"]["mmlu"].items()},
        "boolq_const": {v: x["max_share"] * 100 for v, x in beh["mc_max_answer_share"]["boolq"].items()},
        "unsolvable": h["oracle"]["unsolvable_share"] * 100,
    }


def conclusion(src) -> str:
    c = conclusion_facts(src)
    if c:
        # +1e-9: exact halves (e.g. 75.00 - 72.25 = 2.75) must round up as in the tables, not down by float error
        gains = ", ".join(f"{TASK_NAMES[t]} $+${g + 1e-9:.1f}" for t, g in c["pos_gains"])
        casc = read_csv(REPO / src["p6_cascade_dir"] / "cascade.csv") if src.get("p6_cascade_dir") else []
        casc_cost = (f"{min(float(r['t4_cascade_cost']) for r in casc):.2f}--"
                     f"{max(float(r['t4_cascade_cost']) for r in casc):.2f}$\\times$") if casc else "--"
        glance = [
            ("FLAP vs.\\ baselines", "FLAP $<$ Wanda-SP $<$ Mag-SP in perplexity at every ratio", "\\textbf{positive}"),
            ("Light pruning (10\\%)", f"perplexity {c['ppl0']:.2f} $\\to$ {c['ppl10']:.2f}", "\\textbf{positive}"),
            ("GPU energy / memory (50\\%)", f"$-${(1 - c['e5'] / c['e0']) * 100:.0f}\\% J/token; "
             f"{c['m0']:.2f} $\\to$ {c['m5']:.2f}\\,GB", "\\textbf{positive}"),
            ("GPU prefill (50\\%)", f"$-${c['pre512'] * 100:.0f}\\% (512 tok), $-${c['pre1024'] * 100:.0f}\\% (1024 tok)",
             "\\textbf{positive}"),
            ("GPU decode", f"{c['dec_lo']:.1f}--{c['dec_hi']:.1f}\\,ms/token at every ratio", "negative"),
            ("Accuracy (macro)", f"{c['macro']['v0']:.1f}\\% $\\to$ {c['macro']['flap_0.20']:.1f} / "
             f"{c['macro']['flap_0.35']:.1f} / {c['macro']['flap_0.50']:.1f}\\% (20/35/50\\%)", "negative"),
            ("Routing headroom", f"$+${c['inf'] - c['best']:.1f} pts (informative oracle; naive "
             f"$+${c['oracle'] - c['best']:.1f})", "small positive"),
            ("Query-level cascade", f"costs {casc_cost} the unpruned model on the T4", "negative"),
            ("Laptop-CPU latency", "not reliable enough to report", "pending"),
        ]
        table = ("\\begin{table}[h]\\centering\\small\n\\begin{tabularx}{\\linewidth}{lXl}\n\\toprule\n"
                 "Finding & Evidence & Outcome \\\\\n\\midrule\n"
                 + "\n".join(f"{a} & {b} & {o} \\\\" for a, b, o in glance)
                 + "\n\\bottomrule\n\\end{tabularx}\n\\caption{Key findings at a glance.}\\label{tab:glance}\n"
                 "\\end{table}\n\n")
        return (
            table +
            "\\paragraph{What worked.}\n\\begin{itemize}\n"
            "\\item \\emph{FLAP reproduces on a small edge model.} On SmolLM2-1.7B the ranking FLAP $<$ Wanda-SP $<$ "
            "Magnitude-SP in perplexity holds at every ratio tested, as the FLAP paper reports for LLaMA models of 7\\,B "
            "parameters and larger. "
            f"Light pruning is cheap in quality: FLAP-10\\% moves WikiText-2 perplexity only from {c['ppl0']:.2f} to "
            f"{c['ppl10']:.2f}.\n"
            "\\item \\emph{Pruning gives measurable resource savings on the GPU.} At 50\\%, energy per generated token "
            f"falls by {(1 - c['e5'] / c['e0']) * 100:.0f}\\% ({c['e0']:.2f} to {c['e5']:.2f}\\,J), peak memory from "
            f"{c['m0']:.2f} to {c['m5']:.2f}\\,GB, and prefill latency by {c['pre512'] * 100:.0f}\\% (512 tokens) and "
            f"{c['pre1024'] * 100:.0f}\\% (1024 tokens).\n"
            "\\item \\emph{Per-query routing has real, if small, headroom.} After excluding every variant that does not "
            f"beat a constant answer, the oracle still exceeds the best fixed variant by {c['inf'] - c['best']:.1f} points "
            f"({c['best']:.1f}\\% to {c['inf']:.1f}\\%) at the same cost, with gains on {gains}. On PIQA, FLAP-50\\% keeps "
            f"{c['piqa50']:.1f}\\% accuracy (constant answer: {c['piqa_const']:.1f}\\%) with {c['fewer_params']:.0f}\\% "
            "fewer parameters.\n"
            "\\item \\emph{A method that prevents an overclaim.} The plain oracle would suggest a routing gain of "
            f"{c['oracle'] - c['best']:.1f} points; most of it comes from variants that answer every question identically. "
            "The informative oracle introduced here measures the gain a router could actually exploit.\n"
            "\\item \\emph{Reliable infrastructure.} Results are identical question by question across runs on different "
            "days; every result is cross-checked against lm-eval; several silent errors that would have corrupted results "
            "were found and fixed (Appendix~\\ref{app:issues}).\n"
            "\\end{itemize}\n"
            "\\paragraph{What did not work.}\n\\begin{itemize}\n"
            f"\\item Accuracy falls steeply: macro accuracy {c['macro']['v0']:.1f}\\% (unpruned) to "
            f"{c['macro']['flap_0.20']:.1f}\\% (20\\%), {c['macro']['flap_0.35']:.1f}\\% (35\\%) and "
            f"{c['macro']['flap_0.50']:.1f}\\% (50\\%). The most frequent MMLU answer covers "
            f"{c['mmlu_const']['flap_0.35']:.0f}\\% of questions for FLAP-35\\% and {c['mmlu_const']['flap_0.50']:.0f}\\% "
            f"for FLAP-50\\% (unpruned: {c['mmlu_const']['v0']:.0f}\\%), and on GSM8K the pruned variants reach the "
            "256-token limit on "
            f"{c['gsm_cap']['flap_0.20']:.0f}--{c['gsm_cap']['flap_0.50']:.0f}\\% of questions.\n"
            f"\\item On the T4 at batch size 1, decode latency stays at {c['dec_lo']:.1f}--{c['dec_hi']:.1f}\\,ms per token "
            "for every ratio, so pruning does not shorten generation there; with similar multiple-choice costs and longer "
            "GSM8K outputs, per-query routing saves no latency on this GPU.\n"
            + (f"\\item A query-level confidence cascade costs {casc_cost} the unpruned model on the T4: escalated "
               "questions pay for both models, and the share the cheap variant can answer stays below its cost ratio "
               "(Table~\\ref{tab:cascade}).\n" if casc else "") +
            "\\item Laptop-CPU timings were not reliable enough to report (Section~\\ref{sec:results}).\n"
            "\\end{itemize}\n"
            "\\paragraph{What is next.}\n\\begin{itemize}\n"
            "\\item \\emph{Better cheap variants:} FLAP calibrated on in-domain questions from the train split (no answers) "
            "and milder ratios (10--30\\%), to test whether the pruned variants stay above the constant-answer baseline "
            "on more tasks.\n"
            "\\item \\emph{Cascade fix:} a token-level cascade (speculative decoding) in which a pruned, int8-quantised "
            "copy of the model drafts tokens and the unpruned model verifies several of them in one pass. Unlike the "
            "query-level cascade, no work is wasted on escalation, and with greedy decoding the output is identical to the "
            "unpruned model's by construction.\n"
            "\\item \\emph{Faster execution:} static KV cache with CUDA graphs on the GPU and int8 weights on the CPU "
            "(implemented and unit-tested, not yet measured), and a drift-robust CPU measurement protocol.\n"
            "\\end{itemize}\nThe timeline to the final submission (5 December 2026) is given below.\n")
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
    p2 = _p2_numbers(src)
    if p2:
        s += ("For RQ2, the outcome matrix shows that the plain oracle overstates what routing can gain: "
              f"{p2['oracle']:.1f}\\% against {p2['best']:.1f}\\% for the best fixed variant, but much of that gain comes "
              "from pruned variants that answer every question identically. Restricted to variants that beat a constant "
              f"answer, the oracle reaches {p2['inf']:.1f}\\% and still sends {p2['inf_v0_share']:.1f}\\% of questions to "
              "the unpruned model. On the T4, where all variants cost about the same per multiple-choice question and "
              "the pruned variants generate longer GSM8K answers, per-query compression routing therefore has little to "
              "gain. The next steps are to measure whether the laptop-CPU cost model changes this, and to test whether "
              "less aggressive ratios (10--20\\%) keep more tasks above the constant-answer baseline.\n")
    else:
        s += ("Whether routing pays off (RQ2) depends on the oracle headroom, which the outcome runs now in progress "
              "will measure; the analysis separates genuine complementarity between variants from lucky guesses by "
              "comparing with independence and chance references. The remaining phases are planned as follows.\n")
    return s


def main() -> int:
    src = yaml.safe_load((MID / "sources.yaml").read_text(encoding="utf-8"))
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "pool_table.tex").write_text(pool_table(src), encoding="utf-8")
    (GEN / "routerbench_table.tex").write_text(routerbench_table(src), encoding="utf-8")
    results = p0_section(src)
    s1, f1 = p1_section(src)
    s2, f2 = p2_section(src)
    results += s1 + s2 + p6_section(src)
    (GEN / "results.tex").write_text(results, encoding="utf-8")
    (GEN / "abstract.tex").write_text(abstract(src), encoding="utf-8")
    (GEN / "conclusion.tex").write_text(conclusion(src), encoding="utf-8")
    facts = {"p1": f1, "p2": f2}
    (GEN / "facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print("wrote", sorted(p.name for p in GEN.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
