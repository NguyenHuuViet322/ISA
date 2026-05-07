"""
So sánh ISA vs Hybrid Tabu cho bộ test bé.
- Nhóm theo (Machs, Jobs)
- Mỗi nhóm → 1 figure riêng
- Mỗi figure có 3 subplots: Avg Objective | Best Objective | Box Plot (scatter)

Chạy:
    python plot_comparison_small.py --isa batch_results_ISA_2.csv --tabu tabu_results.csv --out plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot ISA vs Hybrid Tabu (small instances)")
parser.add_argument("--isa",  default="batch_results_ISA_2.csv")
parser.add_argument("--tabu", default="tabu_results.csv")
parser.add_argument("--out",  default="plots")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
isa_raw  = pd.read_csv(args.isa)
tabu_raw = pd.read_csv(args.tabu)
isa_raw.columns  = isa_raw.columns.str.strip()
tabu_raw.columns = tabu_raw.columns.str.strip()

# ISA có cột Machs, Jobs — dùng để nhóm
# Tabu không có, nên join từ ISA
instance_meta = (
    isa_raw[["Instance", "Machs", "Jobs"]]
    .drop_duplicates()
    .sort_values("Instance")
)

# Merge metadata vào tabu
tabu_raw = tabu_raw.merge(instance_meta, on="Instance", how="left")

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
    mpatches.Patch(color=COLOR_ISA,  label="ISA"),
    mpatches.Patch(color=COLOR_TABU, label="Hybrid Tabu"),
]

def agg(df, inst_list):
    rows = []
    for inst in inst_list:
        d = df[df["Instance"] == inst]["Objective"]
        rows.append({
            "instance": inst,
            "obj_avg":  d.mean(),
            "obj_std":  d.std(ddof=1) if len(d) > 1 else 0.0,
            "obj_best": d.min(),
            "values":   d.values,
        })
    return rows

# ── Group by (Machs, Jobs) ────────────────────────────────────────────────────
groups = (
    instance_meta
    .groupby(["Machs", "Jobs"])["Instance"]
    .apply(sorted)
    .reset_index()
)

for _, row in groups.iterrows():
    machs, jobs = int(row["Machs"]), int(row["Jobs"])
    instances   = row["Instance"]

    isa_g  = isa_raw[isa_raw["Instance"].isin(instances)]
    tabu_g = tabu_raw[tabu_raw["Instance"].isin(instances)]

    # Keep only instances present in both
    common = sorted(set(isa_g["Instance"].unique()) & set(tabu_g["Instance"].unique()))
    if not common:
        print(f"  Skipping Machs={machs} Jobs={jobs}: no common instances")
        continue

    N = len(common)
    x = np.arange(N)
    w = 0.35

    isa_rows  = agg(isa_g,  common)
    tabu_rows = agg(tabu_g, common)
    inst_labels = [str(r["instance"]) for r in isa_rows]

    fig, axes = plt.subplots(1, 3, figsize=(max(12, N * 0.9 + 4), 5))
    fig.suptitle(
        f"ISA vs Hybrid Tabu  —  Machs={machs}, Jobs={jobs}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # ── Subplot 1: Average Objective ─────────────────────────────────────────
    ax1 = axes[0]
    isa_avgs  = [r["obj_avg"]  for r in isa_rows]
    tabu_avgs = [r["obj_avg"]  for r in tabu_rows]
    isa_stds  = [r["obj_std"]  for r in isa_rows]
    tabu_stds = [r["obj_std"]  for r in tabu_rows]

    ax1.bar(x - w/2, isa_avgs,  w, yerr=isa_stds,  capsize=4,
            color=COLOR_ISA,  edgecolor="white", linewidth=0.8)
    ax1.bar(x + w/2, tabu_avgs, w, yerr=tabu_stds, capsize=4,
            color=COLOR_TABU, edgecolor="white", linewidth=0.8)

    ax1.set_title("Average Objective Value\n(avg ± std)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Inst {l}" for l in inst_labels], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Objective Value", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_val(v)))
    ax1.yaxis.grid(True, color=GRID_COLOR); ax1.xaxis.grid(False)
    ax1.legend(handles=leg_handles, fontsize=9)

    # ── Subplot 2: Best Objective ─────────────────────────────────────────────
    ax2 = axes[1]
    isa_bests  = [r["obj_best"] for r in isa_rows]
    tabu_bests = [r["obj_best"] for r in tabu_rows]

    ax2.bar(x - w/2, isa_bests,  w,
            color=COLOR_ISA,  edgecolor="white", linewidth=0.8)
    ax2.bar(x + w/2, tabu_bests, w,
            color=COLOR_TABU, edgecolor="white", linewidth=0.8)

    for i, (ib, tb) in enumerate(zip(isa_bests, tabu_bests)):
        if ib == 0:
            continue
        pct  = (ib - tb) / ib * 100
        sign = "+" if pct >= 0 else ""
        top  = max(ib, tb) * 1.01
        ax2.annotate(
            f"{sign}{pct:.1f}%",
            xy=(x[i], top), ha="center", fontsize=7,
            color=COLOR_TABU if pct >= 0 else COLOR_ISA,
        )

    ax2.set_title("Best Objective Value\n(+% = Tabu better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Inst {l}" for l in inst_labels], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Objective Value", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_val(v)))
    ax2.yaxis.grid(True, color=GRID_COLOR); ax2.xaxis.grid(False)
    ax2.legend(handles=leg_handles, fontsize=9)

    # ── Subplot 3: Box + Scatter ──────────────────────────────────────────────
    ax3 = axes[2]

    for i, (ir, tr) in enumerate(zip(isa_rows, tabu_rows)):
        for vals, pos, color in [
            (ir["values"], x[i] - w/2, COLOR_ISA),
            (tr["values"], x[i] + w/2, COLOR_TABU),
        ]:
            if len(vals) == 0:
                continue
            ax3.boxplot(
                vals, positions=[pos], widths=w * 0.75,
                patch_artist=True, notch=False,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker="", linestyle="none"),
            )
            jitter = np.random.uniform(-w * 0.15, w * 0.15, size=len(vals))
            ax3.scatter(
                pos + jitter, vals,
                color=color, s=18, alpha=0.7, zorder=5,
                edgecolors="white", linewidths=0.4,
            )

    ax3.set_title("Objective Distribution\n(box + scatter per run)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Inst {l}" for l in inst_labels], rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Objective Value", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_val(v)))
    ax3.yaxis.grid(True, color=GRID_COLOR); ax3.xaxis.grid(False)
    ax3.legend(handles=leg_handles, fontsize=9)

    fig.tight_layout()

    tag = f"m{machs}_j{jobs}"
    out_png = os.path.join(args.out, f"comparison_{tag}.png")
    out_pdf = os.path.join(args.out, f"comparison_{tag}.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}  /  {out_pdf}")
    plt.close(fig)

print("Done.")