"""
So sánh ISA (STANDARD) vs Hybrid Tabu.
- Mỗi Folder → 1 figure riêng (3 figures)
- Mỗi figure có 3 subplots: Avg Objective | Best Objective | Box Plot (scatter)

Chạy:
    python plot_comparison.py --isa isa_results.csv --tabu tabu_results.csv

Yêu cầu: pandas, matplotlib, numpy
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--isa",  default="o3_results.csv")
parser.add_argument("--tabu", default="tabu_results_big.csv")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
isa_raw  = pd.read_csv(args.isa)
tabu_raw = pd.read_csv(args.tabu)
isa_raw.columns  = isa_raw.columns.str.strip()
tabu_raw.columns = tabu_raw.columns.str.strip()

# Chỉ lấy STANDARD từ ISA
isa_raw = isa_raw[isa_raw["Mode"].str.strip().str.upper() == "STANDARD"].copy()

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_ISA  = "#2C6FAC"
COLOR_TABU = "#E05B3A"
GRID_COLOR = "#E8E8E8"

plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.axisbelow":    True,
})

def fmt_val(v):
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}k"
    return str(int(v))

leg_handles = [
    mpatches.Patch(color=COLOR_ISA,  label="ISA (Standard)"),
    mpatches.Patch(color=COLOR_TABU, label="Hybrid Tabu"),
]

# ── Per-folder loop ───────────────────────────────────────────────────────────
folders = sorted(set(isa_raw["Folder"].unique()) & set(tabu_raw["Folder"].unique()))

for folder in folders:
    isa_f  = isa_raw[isa_raw["Folder"]   == folder].copy()
    tabu_f = tabu_raw[tabu_raw["Folder"] == folder].copy()

    instances = sorted(set(isa_f["Instance"].unique()) & set(tabu_f["Instance"].unique()))
    N = len(instances)
    x = np.arange(N)
    w = 0.35

    # Aggregate
    def agg(df, inst_list):
        rows = []
        for inst in inst_list:
            d = df[df["Instance"] == inst]["Objective"]
            rows.append({
                "instance": inst,
                "obj_avg": d.mean(),
                "obj_std": d.std(ddof=1),
                "obj_best": d.min(),
                "values": d.values,
            })
        return rows

    isa_rows  = agg(isa_f,  instances)
    tabu_rows = agg(tabu_f, instances)

    fig, axes = plt.subplots(1, 3, figsize=(max(12, N * 0.7 + 4), 5))
    fig.suptitle(f"ISA (Standard) vs Hybrid Tabu  —  Folder: {folder}",
                 fontsize=14, fontweight="bold", y=1.02)

    inst_labels = [r["instance"] for r in isa_rows]

    # ── Subplot 1: Average Objective ─────────────────────────────────────────
    ax1 = axes[0]
    isa_avgs  = [r["obj_avg"] for r in isa_rows]
    tabu_avgs = [r["obj_avg"] for r in tabu_rows]
    isa_stds  = [r["obj_std"] for r in isa_rows]
    tabu_stds = [r["obj_std"] for r in tabu_rows]

    ax1.bar(x - w/2, isa_avgs,  w, yerr=isa_stds,  capsize=4,
            color=COLOR_ISA,  edgecolor="white", linewidth=0.8, label="ISA")
    ax1.bar(x + w/2, tabu_avgs, w, yerr=tabu_stds, capsize=4,
            color=COLOR_TABU, edgecolor="white", linewidth=0.8, label="Hybrid Tabu")

    ax1.set_title("Average Objective Value\n(avg ± std)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(inst_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Objective Value", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: fmt_val(v)))
    ax1.yaxis.grid(True, color=GRID_COLOR); ax1.xaxis.grid(False)
    ax1.legend(handles=leg_handles, fontsize=9)

    # ── Subplot 2: Best Objective ─────────────────────────────────────────────
    ax2 = axes[1]
    isa_bests  = [r["obj_best"] for r in isa_rows]
    tabu_bests = [r["obj_best"] for r in tabu_rows]

    b1 = ax2.bar(x - w/2, isa_bests,  w,
                 color=COLOR_ISA,  edgecolor="white", linewidth=0.8)
    b2 = ax2.bar(x + w/2, tabu_bests, w,
                 color=COLOR_TABU, edgecolor="white", linewidth=0.8)

    # % diff annotation
    for i, (ib, tb) in enumerate(zip(isa_bests, tabu_bests)):
        pct = (ib - tb) / ib * 100
        sign = "+" if pct >= 0 else ""
        top = max(ib, tb) * 1.01
        ax2.annotate(f"{sign}{pct:.1f}%",
                     xy=(x[i], top), ha="center", fontsize=7,
                     color=COLOR_TABU if pct >= 0 else COLOR_ISA)

    ax2.set_title("Best Objective Value\n(+% = Tabu better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(inst_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Objective Value", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: fmt_val(v)))
    ax2.yaxis.grid(True, color=GRID_COLOR); ax2.xaxis.grid(False)
    ax2.legend(handles=leg_handles, fontsize=9)

    # ── Subplot 3: Box + Scatter ──────────────────────────────────────────────
    ax3 = axes[2]

    for i, (ir, tr) in enumerate(zip(isa_rows, tabu_rows)):
        pos_isa  = x[i] - w/2
        pos_tabu = x[i] + w/2

        for vals, pos, color in [(ir["values"], pos_isa, COLOR_ISA),
                                  (tr["values"], pos_tabu, COLOR_TABU)]:
            # Box
            bp = ax3.boxplot(vals, positions=[pos], widths=w * 0.75,
                             patch_artist=True, notch=False,
                             boxprops=dict(facecolor=color, alpha=0.5),
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color=color, linewidth=1.2),
                             capprops=dict(color=color, linewidth=1.2),
                             flierprops=dict(marker="", linestyle="none"))
            # Scatter jitter
            jitter = np.random.uniform(-w * 0.15, w * 0.15, size=len(vals))
            ax3.scatter(pos + jitter, vals,
                        color=color, s=18, alpha=0.7, zorder=5, edgecolors="white", linewidths=0.4)

    ax3.set_title("Objective Distribution\n(box + scatter per run)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(inst_labels, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Objective Value", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: fmt_val(v)))
    ax3.yaxis.grid(True, color=GRID_COLOR); ax3.xaxis.grid(False)
    ax3.legend(handles=leg_handles, fontsize=9)

    fig.tight_layout()
    out_png = f"comparison_{folder}.png"
    out_pdf = f"comparison_{folder}.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}  /  {out_pdf}")
    plt.close(fig)

print("Done.")