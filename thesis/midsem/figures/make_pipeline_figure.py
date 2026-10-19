import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys

STY = {
    "input":  ("#FCE9DA", "#E08A3C"),
    "phase":  ("#DEE6F6", "#4472C4"),
    "output": ("#EAF2E6", "#5FA04E"),
    "gate":   ("#FFF4CC", "#C9A100"),
    "opt":    ("#F2E4F5", "#8E44AD"),
    "final":  ("#FDF1CC", "#BF9000"),
}
COL = {"e": 2.5, "a": 8.1, "b": 14.1, "c": 20.1}
W = 5.2
ROW = {1: 13.6, 2: 11.1, 3: 8.6, 4: 6.1, 5: 3.6, 6: 1.2}
H = 1.8

fig, ax = plt.subplots(figsize=(17, 12.5), dpi=150)
ax.set_xlim(-0.2, 25.2); ax.set_ylim(0, 15.6); ax.axis("off")
ax.text(12.5, 15.2, "Project Pipeline (v2): Phase Dependencies & Data Flow",
        ha="center", va="center", fontsize=19, weight="bold")

def box(cx, cy, title, body, kind, w=W, h=H):
    fc, ec = STY[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=2, ls="--" if kind == "opt" else "-"))
    nl = body.count("\n") if body else 0
    if title and body:
        ax.text(cx, cy + 0.38 + 0.12 * max(0, nl - 1), title, ha="center", va="center", fontsize=12.5, weight="bold")
        ax.text(cx, cy - 0.3 - 0.1 * max(0, nl - 1), body, ha="center", va="center", fontsize=10.2, linespacing=1.25)
    elif title:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=12.5, weight="bold")
    else:
        ax.text(cx, cy, body, ha="center", va="center", fontsize=10.2, linespacing=1.25)
    return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, cx=cx, cy=cy)

def arrow(pts, label=None, lpos=None, color="#333333", ls="-", ha="left"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=1.8, ls=ls, zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=18))
    if label:
        ax.text(lpos[0], lpos[1], label, fontsize=10, style="italic", color="#444444", ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=1.2))

base  = box(COL["a"], ROW[1], "Base LLM", "SmolLM2-1.7B\n(LLaMA arch, full MHA)", "input")
bench = box(COL["b"], ROW[1], "Benchmarks (lm-eval)", "MMLU, HellaSwag, WinoGrande, ARC,\nGSM8K, PIQA, BoolQ\n+ RouterBench labels", "input")
edge  = box(COL["e"], ROW[2], "Edge devices", "Laptop CPU, Raspberry Pi 5\nor Jetson (+ Colab T4)", "input", w=4.2)

p1 = box(COL["a"], ROW[2], "PHASE 1 (Wk 1-4)", "FLAP port, ratio sweep,\nlatency/energy profiling", "phase")
p2 = box(COL["b"], ROW[2], "PHASE 2 (Wk 4-7)", "Run every variant on\nevery query (once)", "phase")
o1 = box(COL["a"], ROW[3], None, "Output: nested variant ladder\n(one weight copy) + cost\nmodel per device", "output")
o2 = box(COL["b"], ROW[3], None, "Output: outcome matrix Y[q,v],\ntrain/val/test splits,\nheadroom report", "output")
gate = box(COL["c"], ROW[3], "Decision gate (RQ2)", "Oracle headroom over static\nhull >= ~3 pts? No -> revisit P1", "gate")

p4 = box(COL["a"], ROW[4], "PHASE 4 (Wk 11-14)", "Nested runtime +\nPipeline.run(query)", "phase")
p3 = box(COL["b"], ROW[4], "PHASE 3 (Wk 8-10)", "Cost-aware routers +\nbaselines (offline, CPU)", "phase")

p5 = box((COL["a"] + COL["b"]) / 2, ROW[5], "PHASE 5 (Wk 15-18)",
         "Evaluation: accuracy-cost curves, iso-accuracy savings,\ndeployment modes, GPU vs edge", "phase", w=11.2)
p6 = box(COL["c"], ROW[5], "PHASE 6 (Wk 19-20)", "Optional, if ahead: self-probe\ncascade (RAP-style comparison)", "opt")
p7 = box((COL["a"] + COL["b"]) / 2, ROW[6], "PHASE 7 (Wk 20-22)", "Ablations + thesis writing", "final", w=11.2)

arrow([(base["cx"], base["b"]), (p1["cx"], p1["t"])])
arrow([(bench["cx"], bench["b"]), (p2["cx"], p2["t"])])
arrow([(p1["cx"], p1["b"]), (o1["cx"], o1["t"])])
arrow([(p2["cx"], p2["b"]), (o2["cx"], o2["t"])])
arrow([(p1["r"], p1["cy"]), (p2["l"], p2["cy"])], "variants", ((p1["r"] + p2["l"]) / 2, p1["cy"] + 0.3), ha="center")
arrow([(edge["r"], edge["cy"]), (p1["l"], p1["cy"])], "profiling", ((edge["r"] + p1["l"]) / 2, edge["cy"] + 0.3), ha="center")
arrow([(o2["r"], o2["cy"]), (gate["l"], gate["cy"])])
arrow([(gate["cx"], gate["b"]), (gate["cx"], p3["cy"]), (p3["r"], p3["cy"])], "yes", (gate["cx"] + 0.15, (gate["b"] + p3["cy"]) / 2))
arrow([(o2["cx"], o2["b"]), (p3["cx"], p3["t"])], "train / val", (o2["cx"] + 0.15, (o2["b"] + p3["t"]) / 2))
arrow([(o1["cx"], o1["b"]), (p4["cx"], p4["t"])])
arrow([(p3["l"], p3["cy"]), (p4["r"], p4["cy"])], "router", ((p3["l"] + p4["r"]) / 2, p4["cy"] + 0.3), ha="center")
arrow([(p4["cx"], p4["b"]), (p4["cx"], p5["t"])], "full system", (p4["cx"] + 0.15, (p4["b"] + p5["t"]) / 2))
arrow([(p3["cx"], p3["b"]), (p3["cx"], p5["t"])], "test split", (p3["cx"] + 0.15, (p3["b"] + p5["t"]) / 2))
arrow([(edge["cx"], edge["b"]), (edge["cx"], p5["cy"]), (p5["l"], p5["cy"])], "live edge runs", (edge["cx"] + 0.15, ROW[4]))
arrow([(p5["r"], p5["cy"]), (p6["l"], p6["cy"])], color="#8E44AD", ls="--")
arrow([(p5["cx"], p5["b"]), (p7["cx"], p7["t"])])
arrow([(p6["cx"], p6["b"]), (p6["cx"], p7["cy"]), (p7["r"], p7["cy"])], color="#8E44AD", ls="--")

plt.savefig(sys.argv[1], bbox_inches="tight", facecolor="white")
