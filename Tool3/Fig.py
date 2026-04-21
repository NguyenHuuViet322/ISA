"""
plot_results.py  —  generate all paper figures from comparison CSV
------------------------------------------------------------------
Usage:
    python plot_results.py                      # finds comparison_new.csv in cwd
    python plot_results.py path/to/file.csv

Supports two column naming conventions:
  • TabuO3 vs TabuO3Smart  (from comparision.py output)
  • ISA vs HybridTabu      (from comparison_new.csv)

Output: ./figures/  (PDF + PNG for each figure)
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

C_A   = '#1d9e75'
C_B   = '#d85a30'
C_TIE = '#b4b2a9'
C_BLU = '#378add'

# ── load CSV ──────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'comparison_new.csv'
if not os.path.exists(csv_path):
    cands = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not cands:
        raise FileNotFoundError("No CSV found. Pass path as argument.")
    csv_path = cands[0]

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows  |  columns: {list(df.columns)}\n")

# ── flexible column detection ─────────────────────────────────────
def fc(df, *kws):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in kws):
            return c
    return None

COL = {
    'verdict_avg':  fc(df,'verdict','avg'),
    'verdict_best': fc(df,'verdict','best'),
    'delta_avg':    fc(df,'delta','avg','pct'),
    'delta_best':   fc(df,'delta','best','pct'),
    'a_avg':  (fc(df,'tabuo3_obj_avg')        or fc(df,'hybridt','avg') or fc(df,'tabu','avg')),
    'b_avg':  (fc(df,'tabuo3smart_obj_avg')    or fc(df,'isa','avg')),
    'a_best': (fc(df,'tabuo3_obj_best')        or fc(df,'hybridt','best')),
    'b_best': (fc(df,'tabuo3smart_obj_best')   or fc(df,'isa','best')),
    'a_iter': (fc(df,'tabuo3_iter')            or fc(df,'hybridt','iter') or fc(df,'tabu','iter')),
    'b_iter': (fc(df,'tabuo3smart_iter')       or fc(df,'isa','iter')),
    'folder': fc(df,'folder'),
}

# detect method names
NAME_A, NAME_B = 'Method A', 'Method B'
for c in df.columns:
    cl = c.lower()
    if 'tabuo3smart' in cl: NAME_A, NAME_B = 'TabuO3', 'TabuO3Smart'; break
    if 'hybridt'     in cl: NAME_A, NAME_B = 'HybridTabu', 'ISA'; break
    if 'tabuo3'      in cl: NAME_A, NAME_B = 'TabuO3', 'ISA'; break

print(f"Method A: {NAME_A}  |  Method B: {NAME_B}")
print(f"Columns:  {COL}\n")

def gs(key):
    c = COL.get(key)
    return df[c] if c and c in df.columns else None

n = len(df)
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ── verdict counting ──────────────────────────────────────────────
def count_v(series):
    if series is None:
        return 0, 0, 0
    na = nt = nb = 0
    a_keys = set(); b_keys = set()
    for v in series:
        vs = str(v).lower()
        if 'tie' in vs:
            nt += 1
        else:
            # heuristic: A wins if name_A substring found and name_B not
            a_hit = any(x in vs for x in [NAME_A.lower(), 'tabuo3 ', 'tabuo3_'])
            b_hit = any(x in vs for x in [NAME_B.lower()])
            if 'smart' in NAME_A.lower():
                a_hit = 'smart' in vs
                b_hit = 'tabuo3' in vs and 'smart' not in vs
            if a_hit and not b_hit:
                na += 1
            elif b_hit and not a_hit:
                nb += 1
            elif 'tabuo3smart' in vs:
                if 'smart' in NAME_A.lower(): na += 1
                else: nb += 1
            elif 'tabuo3' in vs:
                if 'smart' not in NAME_A.lower(): na += 1
                else: nb += 1
            elif 'isa' in vs or 'original' in vs:
                if 'isa' in NAME_B.lower(): nb += 1
                else: na += 1
            else:
                nt += 1
    return na, nt, nb

na_b, nt_b, nb_b = count_v(gs('verdict_best'))
na_a, nt_a, nb_a = count_v(gs('verdict_avg'))

# fallback from delta sign
if na_b + nt_b + nb_b == 0 and gs('delta_best') is not None:
    d = gs('delta_best').dropna()
    na_b=(d<0).sum(); nb_b=(d>0).sum(); nt_b=(d==0).sum()
if na_a + nt_a + nb_a == 0 and gs('delta_avg') is not None:
    d = gs('delta_avg').dropna()
    na_a=(d<0).sum(); nb_a=(d>0).sum(); nt_a=(d==0).sum()

print(f"Best: {NAME_A}={na_b}  Tie={nt_b}  {NAME_B}={nb_b}")
print(f"Avg:  {NAME_A}={na_a}  Tie={nt_a}  {NAME_B}={nb_a}\n")

def save(name):
    for ext in ('pdf','png'):
        plt.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {name}")

# ═══════════════════════════════════════════════════════════════════
# FIG 1  Win–Tie–Loss  stacked horizontal bar
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 2.8))
y = np.arange(2)
h = 0.42
va  = [na_b, na_a]
vt  = [nt_b, nt_a]
vb  = [nb_b, nb_a]
b1 = ax.barh(y, va, height=h, color=C_A)
b2 = ax.barh(y, vt, height=h, left=va, color=C_TIE)
b3 = ax.barh(y, vb, height=h, left=[a+t for a,t in zip(va,vt)], color=C_B)
for bars, vals in [(b1,va),(b2,vt),(b3,vb)]:
    for bar, v in zip(bars, vals):
        if v > n * 0.04:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                    f'{v}\n({v/n*100:.1f}%)', ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(['Best solution','Avg solution'])
ax.set_xlabel('Number of instances')
ax.set_title(f'Win–Tie–Loss: {NAME_A} vs. {NAME_B}  (n = {n})')
ax.legend(handles=[mpatches.Patch(color=C_A,label=f'{NAME_A} better'),
                   mpatches.Patch(color=C_TIE,label='Tie'),
                   mpatches.Patch(color=C_B,label=f'{NAME_B} better')],
          loc='lower right', fontsize=8.5, framealpha=0.85)
ax.set_xlim(0, n)
ax.axvline(n/2, color='gray', lw=0.7, ls='--', alpha=0.4)
plt.tight_layout()
save('fig1_win_tie_loss')

# ═══════════════════════════════════════════════════════════════════
# FIG 2  Gap distribution  violin + box  (best & avg)
# ═══════════════════════════════════════════════════════════════════
if gs('delta_avg') is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, title in [(axes[0],'delta_best','Best solution gap (%)'),
                            (axes[1],'delta_avg', 'Avg solution gap (%)')]:
        raw = gs(key)
        if raw is None:
            ax.text(0.5,0.5,'No data',ha='center',va='center'); continue
        d = raw.dropna()
        lo,hi = np.percentile(d,2), np.percentile(d,98)
        clipped = d.clip(lo, hi)

        vp = ax.violinplot(clipped, positions=[0], widths=0.65,
                           showmedians=False, showextrema=False)
        for pc in vp['bodies']:
            pc.set_facecolor('#b5d4f4'); pc.set_alpha(0.55)
            pc.set_edgecolor(C_BLU); pc.set_linewidth(0.8)

        ax.boxplot(clipped, positions=[0], widths=0.2, patch_artist=True,
                   medianprops=dict(color=C_B, linewidth=2.2),
                   boxprops=dict(facecolor='white', edgecolor=C_BLU, linewidth=1.2),
                   whiskerprops=dict(color=C_BLU, linewidth=1),
                   capprops=dict(color=C_BLU, linewidth=1.5),
                   flierprops=dict(marker='.', color='#999', markersize=2.5, alpha=0.25))

        mean_v = d.mean(); med_v = d.median()
        ax.axhline(0, color=C_B, lw=1.3, ls='--', alpha=0.75)
        ax.axhline(mean_v, color=C_BLU, lw=0.9, ls=':', alpha=0.7)

        ax.annotate(f'Median {med_v:.3f}%', xy=(0.07, med_v),
                    xycoords=('axes fraction','data'), fontsize=8, color=C_B, va='center')
        ax.annotate(f'Mean {mean_v:.3f}%', xy=(0.07, mean_v),
                    xycoords=('axes fraction','data'), fontsize=8, color=C_BLU, va='center')

        nA=(d<0).sum(); nB=(d>0).sum(); nT=(d==0).sum()
        ax.text(0.97,0.97,
                f'n = {len(d)}\n{NAME_A} better: {nA} ({nA/len(d)*100:.1f}%)\n'
                f'Tie: {nT} ({nT/len(d)*100:.1f}%)\n{NAME_B} better: {nB} ({nB/len(d)*100:.1f}%)',
                transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.92))
        ax.text(0.02,0.01,f'clipped to [{lo:.2f}%, {hi:.2f}%] (p2–p98)',
                transform=ax.transAxes, fontsize=7, color='gray', va='bottom')
        ax.set_xticks([])
        ax.set_ylabel(f'({NAME_A} – {NAME_B}) / {NAME_B} × 100%')
        ax.set_title(title)

    plt.suptitle(f'Distribution of relative gap: {NAME_A} vs. {NAME_B}', fontsize=11, y=1.01)
    plt.tight_layout()
    save('fig2_gap_distribution')

# ═══════════════════════════════════════════════════════════════════
# FIG 3  Scatter: B avg vs A avg, colored by verdict
# ═══════════════════════════════════════════════════════════════════
if gs('a_avg') is not None and gs('b_avg') is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    da, db = gs('a_avg'), gs('b_avg')
    verdict = gs('verdict_avg')
    if verdict is not None:
        cs = []
        for v in verdict:
            vs = str(v).lower()
            if 'tie' in vs: cs.append(C_TIE)
            elif 'smart' in NAME_A.lower():
                cs.append(C_A if 'smart' in vs else C_B)
            else:
                cs.append(C_B if ('isa' in vs or 'original' in vs) else C_A)
    else:
        diff = da - db
        cs = [C_A if d<0 else (C_TIE if d==0 else C_B) for d in diff]

    ax.scatter(db, da, c=cs, alpha=0.38, s=14, linewidths=0, zorder=3)
    mn = min(da.min(),db.min())*0.98; mx = max(da.max(),db.max())*1.02
    ax.plot([mn,mx],[mn,mx], 'k--', lw=1, alpha=0.55, label='y = x (equal)')
    ax.set_xlabel(f'{NAME_B} — avg objective value')
    ax.set_ylabel(f'{NAME_A} — avg objective value')
    ax.set_title(f'Solution quality per instance\n({NAME_A} vs. {NAME_B}, avg over runs)')
    ax.legend(handles=[mpatches.Patch(color=C_A,alpha=0.7,label=f'{NAME_A} better ({na_a})'),
                        mpatches.Patch(color=C_TIE,alpha=0.7,label=f'Tie ({nt_a})'),
                        mpatches.Patch(color=C_B,alpha=0.7,label=f'{NAME_B} better ({nb_a})'),
                        plt.Line2D([0],[0],c='k',ls='--',lw=1,label='y = x')],
              fontsize=8, framealpha=0.85)
    ax.text(0.02,0.97,f'Below y=x: {NAME_A} better',
            transform=ax.transAxes, fontsize=8, va='top', color=C_A)
    plt.tight_layout()
    save('fig3_scatter')

# ═══════════════════════════════════════════════════════════════════
# FIG 4  Iterations: mean bar + per-instance scatter
# ═══════════════════════════════════════════════════════════════════
if gs('a_iter') is not None and gs('b_iter') is not None:
    ia, ib = gs('a_iter').dropna(), gs('b_iter').dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    means = [ib.mean(), ia.mean()]
    stds  = [ib.std(),  ia.std()]
    bars = ax.bar([NAME_B, NAME_A], means, color=[C_B,C_A], width=0.42, alpha=0.85,
                  yerr=stds, capsize=5, error_kw=dict(elinewidth=1.2,ecolor='#444',capthick=1.2))
    for bar, m, sd in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, m+sd+max(means)*0.015,
                f'{m:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    pct = (ia.mean()-ib.mean())/ib.mean()*100
    ax.set_ylabel('Iterations (avg ± std)')
    ax.set_title('Average iterations within equal time budget')
    ax.text(0.5,0.93,f'{NAME_A}: {pct:+.1f}% vs {NAME_B}',
            transform=ax.transAxes, ha='center', fontsize=9, color=C_A, fontweight='bold')

    ax2 = axes[1]
    mi = min(ia.min(),ib.min()); ma = max(ia.max(),ib.max())
    ax2.scatter(ib, ia, alpha=0.3, s=12, color=C_BLU, linewidths=0)
    ax2.plot([mi,ma],[mi,ma],'k--',lw=1,alpha=0.55,label='y = x')
    ax2.set_xlabel(f'{NAME_B} — iterations')
    ax2.set_ylabel(f'{NAME_A} — iterations')
    ax2.set_title(f'Iterations per instance\n(above y=x: more iters for {NAME_A})')
    ax2.legend(fontsize=8)

    plt.suptitle('Computational effort: search iterations within equal time limit',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save('fig4_iterations')

# ═══════════════════════════════════════════════════════════════════
# FIG 5  Gap by folder / problem family
# ═══════════════════════════════════════════════════════════════════
if COL['folder'] and gs('delta_avg') is not None:
    folders = sorted(df[COL['folder']].dropna().unique())
    if 1 < len(folders) <= 12:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, key, title in [(axes[0],'delta_best','Best solution gap (%) by folder'),
                                (axes[1],'delta_avg', 'Avg solution gap (%) by folder')]:
            groups, labels = [], []
            for fo in folders:
                raw = gs(key)
                if raw is None: continue
                d = raw[df[COL['folder']]==fo].dropna()
                if len(d) < 2: continue
                lo,hi = np.percentile(d,3),np.percentile(d,97)
                groups.append(d.clip(lo,hi).values)
                labels.append(str(fo).replace('_','\n'))
            if groups:
                ax.boxplot(groups, labels=labels, patch_artist=True,
                           medianprops=dict(color=C_B,linewidth=2),
                           boxprops=dict(facecolor='#e6f1fb',edgecolor=C_BLU,linewidth=1),
                           whiskerprops=dict(color=C_BLU,linewidth=0.9),
                           capprops=dict(color=C_BLU,linewidth=1.2),
                           flierprops=dict(marker='.',markersize=2.5,alpha=0.25,color='#999'))
                ax.axhline(0, color=C_B, lw=1.2, ls='--', alpha=0.7)
                ax.set_xlabel('Problem family')
                ax.set_ylabel(f'({NAME_A} – {NAME_B}) / {NAME_B} × 100%')
                ax.set_title(title)
        plt.suptitle(f'Gap distribution by problem family: {NAME_A} vs. {NAME_B}',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        save('fig5_gap_by_folder')

# ═══════════════════════════════════════════════════════════════════
# FIG 6  Improvement magnitude asymmetry
# ═══════════════════════════════════════════════════════════════════
if gs('delta_avg') is not None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, key, title in [(axes[0],'delta_best','Best solution'),
                            (axes[1],'delta_avg', 'Avg solution')]:
        d = gs(key)
        if d is None: continue
        d = d.dropna()
        wins_a = d[d < 0].abs()
        wins_b = d[d > 0]
        if len(wins_a)==0 or len(wins_b)==0: continue
        clip_a = wins_a.quantile(0.97); clip_b = wins_b.quantile(0.97)
        ga = wins_a.clip(0, clip_a).values
        gb = wins_b.clip(0, clip_b).values

        vp = ax.violinplot([ga, gb], positions=[0,1], widths=0.5,
                           showmedians=False, showextrema=False)
        for pc, c in zip(vp['bodies'], [C_A, C_B]):
            pc.set_facecolor(c); pc.set_alpha(0.45)
            pc.set_edgecolor(c); pc.set_linewidth(0.8)

        ax.boxplot([ga, gb], positions=[0,1], widths=0.18, patch_artist=True,
                   medianprops=dict(color='white', linewidth=2),
                   boxprops=dict(facecolor='none', edgecolor='#333', linewidth=1.2),
                   whiskerprops=dict(color='#555', linewidth=0.9),
                   capprops=dict(color='#555', linewidth=1.2),
                   flierprops=dict(marker='.', markersize=2, alpha=0.2))

        ax.set_xticks([0,1])
        ax.set_xticklabels([f'{NAME_A} wins\n(n={len(wins_a)})',
                             f'{NAME_B} wins\n(n={len(wins_b)})'])
        ax.set_ylabel('|Relative improvement| (%)')
        ax.set_title(title)
        for pos, vals, c in [(0, wins_a, C_A),(1, wins_b, C_B)]:
            m = vals.mean()
            ax.text(pos, vals.clip(0, vals.quantile(0.97)).max()*1.1,
                    f'mean\n{m:.3f}%', ha='center', fontsize=8,
                    color=c, fontweight='bold')

    plt.suptitle(f'Improvement magnitude when each method wins: {NAME_A} vs. {NAME_B}',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save('fig6_magnitude')

# ── summary ───────────────────────────────────────────────────────
print(f"\nAll figures saved to ./{OUT}/")
for f in sorted(os.listdir(OUT)):
    kb = os.path.getsize(f'{OUT}/{f}') // 1024
    print(f"  {f:45s}  {kb:4d} KB")