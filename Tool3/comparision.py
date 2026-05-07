"""
compare.py
----------
So sanh ISA vs HybridTabu.

File 1 (ISA):        Instance, Run, Objective, Runtime, Iterations  (tab-separated)
File 2 (HybridTabu): Instance, Run, Objective, Runtime, Iterations  (comma-separated)

Dau ra:
  - comparison.csv : wide format, 1 dong / instance
  - in tom tat ra terminal
"""

from pathlib import Path
import pandas as pd

# ── Duong dan file ────────────────────────────────────────────────────────────
ISA_CSV   = Path("batch_results_ISA_3.csv")
TABU_CSV  = Path("tabu_results.csv")
OUT       = Path("comparison_new.csv")

ALGO_ISA  = "ISA"
ALGO_TABU = "HybridTabu"

# ── Doc file (tu dong detect separator) ──────────────────────────────────────
def read_auto(path: Path) -> pd.DataFrame:
    with open(path) as f:
        header = f.readline()
    sep = "\t" if "\t" in header else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df

isa_raw  = read_auto(ISA_CSV)
tabu_raw = read_auto(TABU_CSV)

# ── Aggregate: best / mean / std theo Instance ────────────────────────────────
def aggregate(df, label):
    def agg(g):
        obj = g["Objective"]
        rt  = g["Runtime"]
        it  = g["Iterations"]
        return pd.Series({
            "obj_best":    obj.min(),
            "obj_avg":     obj.mean(),
            "obj_std":     obj.std(ddof=0),
            "runtime_avg": rt.mean(),
            "iter_avg":    it.mean(),
            "n_runs":      len(obj),
        })
    result = (
        df.groupby("Instance", as_index=False)
          .apply(agg)
          .reset_index(drop=True)
    )
    result.columns = ["Instance"] + [f"{label}_{c}" for c in result.columns if c != "Instance"]
    return result

isa  = aggregate(isa_raw,  ALGO_ISA)
tabu = aggregate(tabu_raw, ALGO_TABU)

# ── Merge ─────────────────────────────────────────────────────────────────────
wide = pd.merge(isa, tabu, on="Instance", how="inner")
wide["Instance"] = wide["Instance"].astype(str)

# ── Delta (Tabu - ISA), negative = Tabu tot hon ───────────────────────────────
wide["delta_best"] = wide[f"{ALGO_TABU}_obj_best"] - wide[f"{ALGO_ISA}_obj_best"]
wide["delta_avg"]  = wide[f"{ALGO_TABU}_obj_avg"]  - wide[f"{ALGO_ISA}_obj_avg"]
wide["delta_best_pct"] = wide["delta_best"] / wide[f"{ALGO_ISA}_obj_best"] * 100
wide["delta_avg_pct"]  = wide["delta_avg"]  / wide[f"{ALGO_ISA}_obj_avg"]  * 100

def verdict(d):
    if pd.isna(d): return "N/A"
    if d < 0:      return "Tabu better"
    if d == 0:     return "Tie"
    return                "ISA better"

wide["verdict_best"] = wide["delta_best"].map(verdict)
wide["verdict_avg"]  = wide["delta_avg"].map(verdict)

# ── Sap xep cot ───────────────────────────────────────────────────────────────
col_order = [
    "Instance",
    f"{ALGO_ISA}_obj_best",  f"{ALGO_TABU}_obj_best",
    "delta_best", "delta_best_pct", "verdict_best",
    f"{ALGO_ISA}_obj_avg",   f"{ALGO_TABU}_obj_avg",
    "delta_avg",  "delta_avg_pct",  "verdict_avg",
    f"{ALGO_ISA}_obj_std",   f"{ALGO_TABU}_obj_std",
    f"{ALGO_ISA}_iter_avg",  f"{ALGO_TABU}_iter_avg",
    f"{ALGO_ISA}_runtime_avg", f"{ALGO_TABU}_runtime_avg",
    f"{ALGO_ISA}_n_runs",    f"{ALGO_TABU}_n_runs",
]
wide = wide[[c for c in col_order if c in wide.columns]]
wide = wide.sort_values("Instance").reset_index(drop=True)
wide.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"Da xuat: {OUT}  (rows={len(wide)})")

# ── Tom tat terminal ──────────────────────────────────────────────────────────
SEP = "=" * 60

def section(label, verdict_col, delta_pct_col):
    total  = len(wide)
    counts = wide[verdict_col].value_counts()
    print(f"\n  [ {label} ]")
    for v in ["Tabu better", "Tie", "ISA better"]:
        n   = counts.get(v, 0)
        pct = n / total * 100
        sub = wide[wide[verdict_col] == v]
        avg_d = sub[delta_pct_col].abs().mean() if len(sub) else float("nan")
        print(f"    {v:<14}: {n:>5}  ({pct:5.1f}%)  avg |delta| = {avg_d:.3f}%")
    overall = wide[delta_pct_col].mean()
    print(f"    Overall delta %       : {overall:+.4f}%")

print(SEP)
print(f"  ISA  vs  HybridTabu")
print(f"  Instances: {len(wide)}")
print(SEP)

section("obj_best", "verdict_best", "delta_best_pct")
section("obj_avg",  "verdict_avg",  "delta_avg_pct")

print(f"\n  [ Runtime & Iterations (avg) ]")
print(f"    ISA       iter: {wide[f'{ALGO_ISA}_iter_avg'].mean():>10.0f}  "
      f"  time: {wide[f'{ALGO_ISA}_runtime_avg'].mean():.3f}s")
print(f"    HybridTabu iter: {wide[f'{ALGO_TABU}_iter_avg'].mean():>10.0f}  "
      f"  time: {wide[f'{ALGO_TABU}_runtime_avg'].mean():.3f}s")

print(SEP)