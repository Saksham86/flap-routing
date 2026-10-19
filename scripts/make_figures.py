"""Report figures, generated from result files only (thesis/midsem/sources.yaml).

fig_ppl.pdf      WikiText-2 perplexity vs. achieved pruning ratio, FLAP / Wanda-SP / Magnitude-SP
fig_t4.pdf       FLAP variants on the T4: prefill latency (3 prompt lengths), decode latency, energy per token
Palette: dataviz reference categorical slots 1-3 (validated: CVD dE 9.2, normal 27.6); every series also has
a distinct marker and a direct label, so identity never depends on colour alone.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MID = REPO / "thesis" / "midsem"
FIG = MID / "figures"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]          # ordinal: 128 / 512 / 1024 tokens

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def rows(sweep_dir: Path) -> list[dict]:
    with (sweep_dir / "sweep.csv").open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def fig_ppl(sweep_dir: Path) -> None:
    rs = rows(sweep_dir)
    base = next(float(r["ppl_wikitext2"]) for r in rs if r["method"] == "none")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    offsets = {"flap": (6, -9, "left"), "wanda_sp": (-6, 11, "right"), "mag_sp": (6, 0, "left")}
    for method, color, marker, name in (("flap", BLUE, "o", "FLAP"), ("wanda_sp", ORANGE, "s", "Wanda-SP"),
                                        ("mag_sp", AQUA, "^", "Magnitude-SP")):
        pts = sorted((float(r["achieved_ratio"]) * 100, float(r["ppl_wikitext2"])) for r in rs if r["method"] == method)
        xs, ys = [0.0] + [p[0] for p in pts], [base] + [p[1] for p in pts]
        ax.plot(xs, ys, color=color, marker=marker, label=name, markeredgecolor="white", markeredgewidth=1)
        dx, dy, ha = offsets[method]
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha,
                    color=INK, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("Prunable parameters removed (%)")
    ax.set_ylabel("WikiText-2 perplexity (log scale)")
    ax.set_xlim(-2, 72)
    ax.axhline(base, color=INK2, linewidth=0.8, linestyle=":")
    ax.annotate(f"unpruned: {base:.2f}", (70, base), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                color=INK2, fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ppl.pdf")
    fig.savefig(FIG / "fig_ppl.png", dpi=200)
    plt.close(fig)


def fig_t4(sweep_dir: Path) -> None:
    prof = json.loads((sweep_dir / "profiles.json").read_text(encoding="utf-8"))
    names = ["none_0.00"] + sorted((k for k in prof if k.startswith("flap_")), key=lambda k: float(k.split("_")[1]))
    x = [0.0 if n.startswith("none") else float(n.split("_")[1]) * 100 for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    ax = axes[0]
    for L, c in zip(("128", "512", "1024"), BLUE_RAMP):
        y = [prof[n]["prefill"][L]["median_ms"] for n in names]
        ax.plot(x, y, color=c, marker="o", markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"{L} tok", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_title("Prefill latency (ms)", fontsize=9, color=INK)
    ax.set_xlim(-3, 75)
    ax.set_ylim(bottom=0)
    ax = axes[1]
    y = [prof[n]["generate"]["decode_ms_per_tok"] for n in names]
    ax.plot(x, y, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=1)
    ax.set_title("Decode latency (ms / token)", fontsize=9, color=INK)
    ax.set_ylim(0, max(y) * 1.25)
    ax = axes[2]
    y = []
    for n in names:
        e = prof[n]["energy"]
        y.append(e["joules_per_generated_token"])
    ax.plot(x, y, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=1)
    ax.set_title("Energy (J / generated token)", fontsize=9, color=INK)
    ax.set_ylim(0, max(y) * 1.25)
    for ax in axes:
        ax.set_xlabel("FLAP ratio (%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_t4.pdf")
    fig.savefig(FIG / "fig_t4.png", dpi=200)
    plt.close(fig)


def main() -> int:
    src = yaml.safe_load((MID / "sources.yaml").read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)
    if src.get("p1_sweep_dir"):
        d = REPO / src["p1_sweep_dir"]
        fig_ppl(d)
        fig_t4(d)
        print("figures written:", sorted(p.name for p in FIG.glob("fig_*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
