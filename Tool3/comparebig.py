"""
compare_o3_tabu.py
------------------
So sanh 3 algo:
  - O3 STANDARD  (file 1, Mode=STANDARD)
  - O3 SMART     (file 1, Mode=SMART)
  - Tabu         (file 2)

File 1: Folder, Instance, Mode, Run, Objective, Runtime, Iterations
File 2: Folder, Instance, Run, Objective, Runtime, Iterations, Intensify, Diversify, Aspiration

Dau ra:
  - summary.csv      : long format, 1 dong / algo x instance
  - comparison.csv   : wide format, 1 dong / instance, de so sanh ngang
  - in tom tat ra terminal
"""

from pathlib import Path
import pandas as pd

# -- Duong dan file -----------------------------------------------------------
O3_CSV   = Path("o3_results.csv")      # file chua ca STANDARD va SMART
TABU_CSV = Path("tabu_results_big.csv")    # file Tabu
OUT_SUMMARY    = Path("summary.csv")
OUT_COMPARISON = Path("comparison.csv")

ALGO_STD  = "O3_Standard"
ALGO_SMRT = "O3_Smart"
ALGO_TABU = "Tabu"

# -- Ham tong hop theo (Folder, Instance) ------------------------------------
def aggregate(df, algo_label):
    def agg(g):
        obj = g["Objective"]
        rt  = g["Runtime"]
        return pd.Series({
            "obj_best":     obj.min(),
            "obj_avg":      obj.mean(),
            "runtime_avg":  rt.mean(),
            "n_runs":       len(obj),
        })

    result = (
        df.groupby(["Folder", "Instance"], as_index=False)
          .apply(agg)
          .reset_index(drop=True)
    )
    result.insert(2, "algo", algo_label)
    return result

# -- Doc & xu ly file O3 (co cot Mode) ---------------------------------------
o3_raw = pd.read_csv(O3_CSV)
o3_raw.columns = [c.strip() for c in o3_raw.columns]

std  = aggregate(o3_raw[o3_raw["Mode"] == "STANDARD"].copy(), ALGO_STD)
smrt = aggregate(o3_raw[o3_raw["Mode"] == "SMART"].copy(),    ALGO_SMRT)

# -- Doc & xu ly file Tabu ----------------------------------------------------
tabu_raw = pd.read_csv(TABU_CSV)
tabu_raw.columns = [c.strip() for c in tabu_raw.columns]

tabu = aggregate(tabu_raw, ALGO_TABU)

# -- Long format: gop 3 algo lai ----------------------------------------------
summary = pd.concat([std, smrt, tabu], ignore_index=True)
summary = summary.sort_values(["Folder", "Instance", "algo"]).reset_index(drop=True)

summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
print(f"Da xuat: {OUT_SUMMARY}  (rows={len(summary)})")

# -- Wide format: pivot de so sanh ngang --------------------------------------
wide = summary.pivot(
    index=["Folder", "Instance"],
    columns="algo",
    values=["obj_best", "obj_avg", "runtime_avg"]
)
wide.columns = [f"{algo}_{stat}" for stat, algo in wide.columns]
wide = wide.reset_index()

# -- Delta: Tabu va Smart so voi Standard (baseline) -------------------------
for algo, label in [(ALGO_SMRT, "Smart"), (ALGO_TABU, "Tabu")]:
    for metric in ["obj_best", "obj_avg"]:
        col_base = f"{ALGO_STD}_{metric}"
        col_cmp  = f"{algo}_{metric}"
        wide[f"delta_{label}_{metric}"] = wide[col_cmp] - wide[col_base]
        wide[f"delta_{label}_{metric}_pct"] = (
            (wide[col_cmp] - wide[col_base]) / wide[col_base] * 100
        )

# Verdict: am = algo do tot hon Standard (minimize)
def verdict_vs_std(d):
    if pd.isna(d):  return "N/A"
    if d < 0:       return "Better"
    if d == 0:      return "Tie"
    return                  "Worse"

for label in ["Smart", "Tabu"]:
    wide[f"verdict_{label}"] = wide[f"delta_{label}_obj_best"].map(verdict_vs_std)

# Sap xep cot
col_order = (
    ["Folder", "Instance"]
    + [f"{ALGO_STD}_obj_best",  f"{ALGO_SMRT}_obj_best",  f"{ALGO_TABU}_obj_best"]
    + [f"delta_Smart_obj_best", f"delta_Smart_obj_best_pct", f"verdict_Smart"]
    + [f"delta_Tabu_obj_best",  f"delta_Tabu_obj_best_pct",  f"verdict_Tabu"]
    + [f"{ALGO_STD}_obj_avg",   f"{ALGO_SMRT}_obj_avg",   f"{ALGO_TABU}_obj_avg"]
    + [f"delta_Smart_obj_avg",  f"delta_Smart_obj_avg_pct"]
    + [f"delta_Tabu_obj_avg",   f"delta_Tabu_obj_avg_pct"]
    + [f"{ALGO_STD}_runtime_avg", f"{ALGO_SMRT}_runtime_avg", f"{ALGO_TABU}_runtime_avg"]
)
wide = wide[[c for c in col_order if c in wide.columns]]
wide.to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")
print(f"Da xuat: {OUT_COMPARISON}  (rows={len(wide)})")

# -- Tom tat terminal ---------------------------------------------------------
SEP = "=" * 65

def print_section(metric_label, delta_col_pct, verdict_col, algo_label):
    total  = wide[verdict_col].notna().sum()
    counts = wide[verdict_col].value_counts()

    better = wide[wide[verdict_col] == "Better"]
    worse  = wide[wide[verdict_col] == "Worse"]

    better_pct = better[delta_col_pct].abs().mean()
    worse_pct  = worse[delta_col_pct].abs().mean()

    print("")
    print(f"  [ {metric_label} ]  (so voi {ALGO_STD} lam baseline)")
    for label in ["Better", "Tie", "Worse"]:
        n   = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        print(f"    {label:<8}: {n:>4}  ({pct:5.1f}%)")
    print(f"    Khi {algo_label} tot hon -> tot hon avg : {better_pct:.2f}%")
    print(f"    Khi {algo_label} kem hon -> kem hon avg : {worse_pct:.2f}%")
    overall = wide[delta_col_pct].mean()
    print(f"    Delta % overall          : {overall:+.2f}%")

print(SEP)
print(f"  O3 Standard  vs  O3 Smart  vs  Tabu")
print(f"  Tong so instances: {len(wide)}")
print(SEP)

print_section("obj_best -- O3 Smart vs Standard",
              "delta_Smart_obj_best_pct", "verdict_Smart", ALGO_SMRT)

print_section("obj_best -- Tabu vs Standard",
              "delta_Tabu_obj_best_pct", "verdict_Tabu", ALGO_TABU)

print("")
print("  [ obj_avg -- Delta overall ]")
print(f"    Smart vs Standard : {wide['delta_Smart_obj_avg_pct'].mean():+.2f}%")
print(f"    Tabu  vs Standard : {wide['delta_Tabu_obj_avg_pct'].mean():+.2f}%")

print("")
print("  [ Runtime avg ]")
for algo, col in [(ALGO_STD,  f"{ALGO_STD}_runtime_avg"),
                  (ALGO_SMRT, f"{ALGO_SMRT}_runtime_avg"),
                  (ALGO_TABU, f"{ALGO_TABU}_runtime_avg")]:
    val = wide[col].mean() if col in wide.columns else float("nan")
    print(f"    {algo:<14}: {val:.3f}s")

print(SEP)