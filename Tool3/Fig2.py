"""
plot_big_test.py  —  figures for 3-method comparison
-----------------------------------------------------
File format (from large benchmark):
    Folder, Instance,
    O3_Standard_obj_best, O3_Smart_obj_best, Tabu_obj_best,
    delta_Smart_obj_best, delta_Smart_obj_best_pct, verdict_Smart,
    delta_Tabu_obj_best,  delta_Tabu_obj_best_pct,  verdict_Tabu,
    O3_Standard_obj_avg,  O3_Smart_obj_avg,  Tabu_obj_avg,
    delta_Smart_obj_avg,  delta_Smart_obj_avg_pct,
    delta_Tabu_obj_avg,   delta_Tabu_obj_avg_pct,
    O3_Standard_runtime_avg, O3_Smart_runtime_avg, Tabu_runtime_avg

delta_X = X - O3_Standard  →  negative = X is BETTER than Standard

Usage:
    python plot_big_test.py                    # looks for big_test_data.csv
    python plot_big_test.py path/to/file.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# ── colors ────────────────────────────────────────────────────────
C_STD   = '#888780'   # O3 Standard  (gray)
C_SMART = '#378add'   # O3 Smart     (blue)
C_TABU  = '#1d9e75'   # Tabu Hybrid  (green)
C_WORSE = '#d85a30'   # worse than standard

# ── load ──────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'comparison.csv'
if not os.path.exists(csv_path):
    cands = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not cands:
        raise FileNotFoundError("No CSV found.")
    csv_path = cands[0]

df = pd.read_csv(csv_path)
n = len(df)
print(f"Loaded {n} rows | {csv_path}")
print(f"Folders: {sorted(df['Folder'].unique())}\n")

OUT = 'figures_big'
os.makedirs(OUT, exist_ok=True)

def save(name):
    for ext in ('pdf', 'png'):
        plt.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved {name}")

# ── helper: verdict counts ─────────────────────────────────────────
def vc(col):
    """Count Better / Worse / (implicit Tie if absent)."""
    better = (df[col] == 'Better').sum()
    worse  = (df[col] == 'Worse').sum()
    tie    = n - better - worse
    return better, tie, worse

smart_b_best, smart_t_best, smart_w_best = vc('verdict_Smart')
tabu_b_best,  tabu_t_best,  tabu_w_best  = vc('verdict_Tabu')

# avg verdict inferred from delta sign
smart_b_avg = (df['delta_Smart_obj_avg'] < 0).sum()
smart_w_avg = (df['delta_Smart_obj_avg'] > 0).sum()
smart_t_avg = (df['delta_Smart_obj_avg'] == 0).sum()

tabu_b_avg  = (df['delta_Tabu_obj_avg'] < 0).sum()
tabu_w_avg  = (df['delta_Tabu_obj_avg'] > 0).sum()
tabu_t_avg  = (df['delta_Tabu_obj_avg'] == 0).sum()

print("Win–Tie–Loss vs O3 Standard:")
print(f"  O3 Smart | Best:  Better={smart_b_best} Tie={smart_t_best} Worse={smart_w_best}")
print(f"  O3 Smart | Avg:   Better={smart_b_avg}  Tie={smart_t_avg}  Worse={smart_w_avg}")
print(f"  Tabu     | Best:  Better={tabu_b_best}  Tie={tabu_t_best}  Worse={tabu_w_best}")
print(f"  Tabu     | Avg:   Better={tabu_b_avg}   Tie={tabu_t_avg}   Worse={tabu_w_avg}")
print()

# ═══════════════════════════════════════════════════════════════════
# FIG 1  Win–Tie–Loss  (4 rows: Smart/best, Smart/avg, Tabu/best, Tabu/avg)
# ═══════════════════════════════════════════════════════════════════
print("Generating figures...")
fig, ax = plt.subplots(figsize=(8, 3.5))

rows = [
    (f'O3 Smart\n(best)',  smart_b_best, smart_t_best, smart_w_best, C_SMART),
    (f'O3 Smart\n(avg)',   smart_b_avg,  smart_t_avg,  smart_w_avg,  C_SMART),
    (f'Tabu Hybrid\n(best)', tabu_b_best, tabu_t_best, tabu_w_best, C_TABU),
    (f'Tabu Hybrid\n(avg)',  tabu_b_avg,  tabu_t_avg,  tabu_w_avg,  C_TABU),
]
y = np.arange(len(rows))
h = 0.38

for i, (label, nb, nt, nw, col) in enumerate(rows):
    left = 0
    for val, fc, lbl in [(nb, col, 'Better'), (nt, '#c8c7c2', 'Tie'), (nw, C_WORSE, 'Worse')]:
        bar = ax.barh(i, val, height=h, left=left, color=fc, alpha=0.88)
        if val > n * 0.04:
            ax.text(left + val/2, i, f'{val}\n({val/n*100:.0f}%)',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
        left += val

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('Number of instances')
ax.set_title(f'Win–Tie–Loss vs. O3 Standard (baseline)  |  n = {n}')
ax.set_xlim(0, n)
ax.axvline(n/2, color='gray', lw=0.7, ls='--', alpha=0.4)
ax.legend(handles=[
    mpatches.Patch(color=C_TABU,   alpha=0.88, label='Better than Standard'),
    mpatches.Patch(color='#c8c7c2', alpha=0.88, label='Tie'),
    mpatches.Patch(color=C_WORSE,  alpha=0.88, label='Worse than Standard'),
], loc='lower right', fontsize=8.5, framealpha=0.85)
plt.tight_layout()
save('fig1_win_tie_loss')

# ═══════════════════════════════════════════════════════════════════
# FIG 2  Gap distribution: Smart vs Standard  &  Tabu vs Standard
#        violin + box, best and avg side by side
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

combos = [
    (axes[0,0], 'delta_Smart_obj_best_pct', 'O3 Smart — best solution gap (%)',  C_SMART),
    (axes[0,1], 'delta_Tabu_obj_best_pct',  'Tabu Hybrid — best solution gap (%)', C_TABU),
    (axes[1,0], 'delta_Smart_obj_avg_pct',  'O3 Smart — avg solution gap (%)',   C_SMART),
    (axes[1,1], 'delta_Tabu_obj_avg_pct',   'Tabu Hybrid — avg solution gap (%)',  C_TABU),
]

for ax, col, title, col_color in combos:
    if col not in df.columns:
        ax.text(0.5, 0.5, f'{col}\nnot found', ha='center', va='center')
        continue
    d = df[col].dropna()
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    clipped = d.clip(lo, hi)

    vp = ax.violinplot(clipped, positions=[0], widths=0.65,
                       showmedians=False, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(col_color); pc.set_alpha(0.35)
        pc.set_edgecolor(col_color); pc.set_linewidth(0.8)

    ax.boxplot(clipped, positions=[0], widths=0.2, patch_artist=True,
               medianprops=dict(color='white', linewidth=2.2),
               boxprops=dict(facecolor=col_color, edgecolor=col_color,
                             linewidth=1.2, alpha=0.7),
               whiskerprops=dict(color=col_color, linewidth=1),
               capprops=dict(color=col_color, linewidth=1.5),
               flierprops=dict(marker='.', color='#999', markersize=2.5, alpha=0.25))

    mean_v = d.mean(); med_v = d.median()
    ax.axhline(0, color='#555', lw=1.2, ls='--', alpha=0.65, label='No difference')
    ax.axhline(mean_v, color=col_color, lw=0.9, ls=':', alpha=0.8)

    # labels
    y_offset = (hi - lo) * 0.06
    ax.text(0.55, med_v,    f'Med {med_v:.2f}%',  transform=ax.get_yaxis_transform(),
            fontsize=8, color='white', va='center', fontweight='bold')
    ax.text(0.55, mean_v,   f'Mean {mean_v:.2f}%', transform=ax.get_yaxis_transform(),
            fontsize=8, color=col_color, va='center')

    n_b = (d < 0).sum(); n_w = (d > 0).sum(); n_t = (d == 0).sum()
    ax.text(0.97, 0.97,
            f'n = {len(d)}\nBetter: {n_b} ({n_b/len(d)*100:.1f}%)\n'
            f'Tie: {n_t}\nWorse: {n_w} ({n_w/len(d)*100:.1f}%)\n'
            f'Mean gap: {mean_v:.2f}%',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.92))
    ax.text(0.02, 0.01, f'clipped [{lo:.1f}%, {hi:.1f}%] (p2–p98)',
            transform=ax.transAxes, fontsize=7, color='gray', va='bottom')

    ax.set_xticks([])
    ax.set_ylabel('(Method – O3 Standard) / O3 Standard × 100%')
    ax.set_title(title)

plt.suptitle('Gap distribution relative to O3 Standard baseline\n'
             '(negative = improvement over Standard)', fontsize=11, y=1.01)
plt.tight_layout()
save('fig2_gap_distribution')

# ═══════════════════════════════════════════════════════════════════
# FIG 3  Side-by-side avg gap per folder  (grouped bar)
# ═══════════════════════════════════════════════════════════════════
folders = sorted(df['Folder'].unique())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, col_smart, col_tabu, metric in [
    (axes[0], 'delta_Smart_obj_best_pct', 'delta_Tabu_obj_best_pct', 'Best solution'),
    (axes[1], 'delta_Smart_obj_avg_pct',  'delta_Tabu_obj_avg_pct',  'Avg solution'),
]:
    x = np.arange(len(folders))
    w = 0.32
    smart_means = [df[df['Folder']==f][col_smart].mean() for f in folders]
    tabu_means  = [df[df['Folder']==f][col_tabu].mean()  for f in folders]
    smart_stds  = [df[df['Folder']==f][col_smart].std()  for f in folders]
    tabu_stds   = [df[df['Folder']==f][col_tabu].std()   for f in folders]

    b1 = ax.bar(x - w/2, smart_means, width=w, color=C_SMART, alpha=0.85,
                yerr=smart_stds, capsize=4,
                error_kw=dict(elinewidth=1, ecolor='#444', capthick=1),
                label='O3 Smart')
    b2 = ax.bar(x + w/2, tabu_means,  width=w, color=C_TABU,  alpha=0.85,
                yerr=tabu_stds, capsize=4,
                error_kw=dict(elinewidth=1, ecolor='#444', capthick=1),
                label='Tabu Hybrid')

    # value labels on bars
    for bars, vals in [(b1, smart_means), (b2, tabu_means)]:
        for bar, v in zip(bars, vals):
            ypos = v + (bar.get_yerr() if hasattr(bar, 'get_yerr') else 0)
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (abs(v)*0.05 if v < 0 else abs(v)*0.05),
                    f'{v:.1f}%', ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=7.5, color='#333')

    ax.axhline(0, color='#555', lw=1, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in folders])
    ax.set_ylabel('Mean gap vs. O3 Standard (%)')
    ax.set_title(f'{metric}\n(negative = better than Standard)')
    ax.legend(fontsize=8.5, framealpha=0.85)

plt.suptitle('Mean improvement over O3 Standard by problem family\n'
             '(error bars = ±1 std)', fontsize=11, y=1.01)
plt.tight_layout()
save('fig3_gap_by_folder')

# ═══════════════════════════════════════════════════════════════════
# FIG 4  Scatter: Standard vs Smart  &  Standard vs Tabu  (avg obj)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, ycol, col, name in [
    (axes[0], 'O3_Smart_obj_avg',  C_SMART, 'O3 Smart'),
    (axes[1], 'Tabu_obj_avg',      C_TABU,  'Tabu Hybrid'),
]:
    x = df['O3_Standard_obj_avg']
    y = df[ycol]

    # color by folder
    folder_list = sorted(df['Folder'].unique())
    folder_colors = ['#378add', '#1d9e75', '#d85a30']
    fc_map = {f: folder_colors[i % len(folder_colors)] for i, f in enumerate(folder_list)}
    cs = [fc_map[f] for f in df['Folder']]

    ax.scatter(x, y, c=cs, alpha=0.7, s=60, linewidths=0.5,
               edgecolors='white', zorder=3)

    mn = min(x.min(), y.min()) * 0.97
    mx = max(x.max(), y.max()) * 1.03
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.2, alpha=0.5, label='y = x (equal)')

    # add folder legend
    handles = [plt.Line2D([0],[0], c='k', ls='--', lw=1.2, label='y = x (equal)')]
    for f in folder_list:
        handles.append(mpatches.Patch(color=fc_map[f], label=f, alpha=0.8))
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.85, loc='upper left')

    ax.set_xlabel('O3 Standard — avg objective')
    ax.set_ylabel(f'{name} — avg objective')
    ax.set_title(f'{name} vs. O3 Standard\n(below y=x: {name} better)')

    below = (y < x).sum()
    ax.text(0.97, 0.05,
            f'Below y=x: {below}/{n} ({below/n*100:.0f}%)',
            transform=ax.transAxes, ha='right', fontsize=8,
            color=col, fontweight='bold')

plt.suptitle('Per-instance solution quality: O3 Smart and Tabu Hybrid vs. O3 Standard\n'
             '(avg objective over all runs)', fontsize=11, y=1.01)
plt.tight_layout()
save('fig4_scatter')

# ═══════════════════════════════════════════════════════════════════
# FIG 5  Cumulative distribution of gap %  (Smart & Tabu, best & avg)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

for ax, pairs, title in [
    (axes[0],
     [('delta_Smart_obj_best_pct', C_SMART, 'O3 Smart (best)'),
      ('delta_Tabu_obj_best_pct',  C_TABU,  'Tabu Hybrid (best)')],
     'Best solution gap — CDF'),
    (axes[1],
     [('delta_Smart_obj_avg_pct', C_SMART, 'O3 Smart (avg)'),
      ('delta_Tabu_obj_avg_pct',  C_TABU,  'Tabu Hybrid (avg)')],
     'Avg solution gap — CDF'),
]:
    for col, color, label in pairs:
        if col not in df.columns: continue
        d = np.sort(df[col].dropna().values)
        cdf = np.arange(1, len(d)+1) / len(d)
        ax.plot(d, cdf, color=color, lw=2, label=label)
        # mark 50th percentile
        med = np.median(d)
        ax.axvline(med, color=color, lw=0.8, ls=':', alpha=0.6)
        ax.text(med, 0.05, f'{med:.1f}%', color=color,
                fontsize=7.5, ha='center', va='bottom')

    ax.axvline(0, color='#555', lw=1, ls='--', alpha=0.55)
    ax.set_xlabel('Gap vs. O3 Standard (%)\n(negative = better than Standard)')
    ax.set_ylabel('Cumulative fraction of instances')
    ax.set_title(title)
    ax.legend(fontsize=8.5, framealpha=0.85)
    # shade the "improvement" region
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < 0 else -50, 0,
               alpha=0.04, color=C_TABU)

plt.suptitle('Cumulative distribution of relative gap vs. O3 Standard\n'
             'Shaded region: improvement over Standard', fontsize=11, y=1.01)
plt.tight_layout()
save('fig5_cdf')

# ═══════════════════════════════════════════════════════════════════
# FIG 6  Improvement magnitude asymmetry per method
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, col_smart, col_tabu, metric in [
    (axes[0], 'delta_Smart_obj_best_pct', 'delta_Tabu_obj_best_pct', 'Best solution'),
    (axes[1], 'delta_Smart_obj_avg_pct',  'delta_Tabu_obj_avg_pct',  'Avg solution'),
]:
    ds = df[col_smart].dropna()
    dt = df[col_tabu].dropna()

    groups = [
        ds[ds < 0].abs().clip(0, ds[ds < 0].abs().quantile(0.97)),
        dt[dt < 0].abs().clip(0, dt[dt < 0].abs().quantile(0.97)),
        ds[ds > 0].clip(0, ds[ds > 0].quantile(0.97)),
        dt[dt > 0].clip(0, dt[dt > 0].quantile(0.97)),
    ]
    colors_v = [C_SMART, C_TABU, '#e8a888', '#a8d4c0']
    positions = [0, 1, 2.5, 3.5]
    labels = ['Smart\nimproves', 'Tabu\nimproves', 'Smart\ndegrades', 'Tabu\ndegrades']

    # filter out empty groups
    valid = [(g, p, c, l) for g, p, c, l in zip(groups, positions, colors_v, labels) if len(g) > 1]
    if not valid: continue
    groups_v, positions_v, colors_v2, labels_v = zip(*valid)

    vp = ax.violinplot([g.values for g in groups_v], positions=list(positions_v),
                       widths=0.55, showmedians=False, showextrema=False)
    for pc, c in zip(vp['bodies'], colors_v2):
        pc.set_facecolor(c); pc.set_alpha(0.45)
        pc.set_edgecolor(c); pc.set_linewidth(0.8)

    ax.boxplot([g.values for g in groups_v], positions=list(positions_v),
               widths=0.18, patch_artist=True,
               medianprops=dict(color='white', linewidth=2),
               boxprops=dict(facecolor='none', edgecolor='#333', linewidth=1.1),
               whiskerprops=dict(color='#555', linewidth=0.9),
               capprops=dict(color='#555', linewidth=1.2),
               flierprops=dict(marker='.', markersize=2, alpha=0.2))

    for pos, g, c in zip(positions_v, groups_v, colors_v2):
        if len(g) == 0: continue
        is_smart = pos in [0, 2]
        src = df[col_smart].dropna() if is_smart else df[col_tabu].dropna()
        sub = src[src < 0].abs() if pos in [0, 1] else src[src > 0]
        if len(sub) > 0:
            ax.text(pos, g.max()*1.15,
                    f'mean\n{sub.mean():.2f}%',
                    ha='center', fontsize=8, color=c, fontweight='bold')

    ax.set_xticks(list(positions_v))
    ax.set_xticklabels([labels[positions.index(p)] for p in positions_v])
    ax.axvline(1.75, color='#ccc', lw=1, ls='-')
    ax.text(0.5, 1.01, 'Improvements', transform=ax.transAxes,
            ha='center', fontsize=9, color='#333', style='italic')
    ax.set_ylabel('|Gap| (%)')
    ax.set_title(metric)

plt.suptitle('Magnitude of improvement vs. degradation relative to O3 Standard',
             fontsize=11, y=1.01)
plt.tight_layout()
save('fig6_magnitude')

# ── summary ───────────────────────────────────────────────────────
print(f"\nAll figures saved to ./{OUT}/")
for f in sorted(os.listdir(OUT)):
    kb = os.path.getsize(f'{OUT}/{f}') // 1024
    print(f"  {f:45s}  {kb:4d} KB")