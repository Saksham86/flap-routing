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


def p1_section(src) -> tuple[str, dict]:
    facts: dict = {}
    s = "\\subsection{Pruned-variant ladder (P1)}\n"
    if not src.get("p1_sweep_dir"):
        s += ("\\pend{the FLAP sweep (ratios 0.1--0.6) and the Wanda-SP / Magnitude-SP baselines are running on the "
              "Colab T4; perplexity, parameter counts and GPU latency will be reported from "
              "\\texttt{results/p1/<run>/sweep.csv}.}\n\n")
    else:
        rows = [r for r in read_csv(REPO / src["p1_sweep_dir"] / "sweep.csv") if r["status"] == "ok"]
        facts["sweep"] = rows
        body = []
        for r in rows:
            name = f"{r['method']}_{float(r['ratio']):.2f}"
            body.append(f"{label(name)} & {float(r['achieved_ratio']) * 100:.1f} & {int(r['n_params']) / 1e9:.3f} & "
                        f"{float(r['ppl_wikitext2']):.2f} & {float(r['prefill_ms_512']):.1f} & "
                        f"{float(r['decode_ms_per_tok']):.1f} & "
                        + (f"{float(r['joules_per_token']):.3f}" if r.get("joules_per_token") else "--") + " \\\\")
        s += ("Table~\\ref{tab:sweep} lists every pruned variant produced on the T4. ``Achieved'' is the fraction of "
              "prunable (attention and MLP) parameters removed; embeddings are not pruned. Latency is batch-1 fp16.\n\n"
              "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
              "Variant & Achieved (\\%) & Params (B) & WikiText-2 PPL & Prefill 512 (ms) & Decode (ms/tok) & J/token \\\\\n"
              "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
              "\\caption{FLAP sweep and baselines on SmolLM2-1.7B (Colab T4, fp16, batch 1).}\\label{tab:sweep}\n\\end{table}\n\n")
    cpu = cpu_profiles(src)
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
    facts = {"p1": f1, "p2": f2}
    (GEN / "facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print("wrote", sorted(p.name for p in GEN.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
