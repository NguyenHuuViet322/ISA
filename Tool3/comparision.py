"""
compare.py
----------
Gop ISA summary + Tabu summary roi so sanh.

Workflow:
  1. python summarize.py       -> summary_obj_by_test.csv
  2. python process_tabu.py    -> tabu_summary.csv
  3. python compare.py         -> comparison.csv + tom tat ra terminal
"""

from pathlib import Path
import pandas as pd

ISA_CSV  = Path("summary_obj_by_test.csv")
TABU_CSV = Path("tabu_summary.csv")
OUT_CSV  = Path("comparison.csv")

ALGO_A = "ISA"
ALGO_B = "Tabu+O3"

# -- Doc ISA: gop TAT CA cac algo (I_1_10, I_11_20...) thanh 1 bang ----------
isa_all = pd.read_csv(ISA_CSV)

isa = (
    isa_all[["test_id", "mach", "job", "lower_cost", "upper_cost",
              "obj_best", "obj_mean", "runtime_mean"]]
    .dropna(subset=["obj_best"])
    .drop_duplicates("test_id")
    .copy()
)
isa["algo"] = ALGO_A

# -- Doc Tabu -----------------------------------------------------------------
tabu = pd.read_csv(TABU_CSV)
tabu["algo"] = ALGO_B

# -- Pivot sang wide de so sanh ngang -----------------------------------------
combined = pd.concat([isa, tabu], ignore_index=True)
combined["test_num"] = combined["test_id"].str.replace("T_", "", regex=False).astype(int)

shared = (
    combined[["test_id", "test_num", "mach", "job", "lower_cost", "upper_cost"]]
    .dropna(subset=["mach"])
    .drop_duplicates("test_id")
    .set_index("test_id")
)

wide = combined.pivot(index="test_id", columns="algo",
                      values=["obj_best", "obj_mean", "runtime_mean"])
wide.columns = [f"{algo}_{stat}" for stat, algo in wide.columns]
wide = shared.join(wide).reset_index()

# -- Cot delta & verdict ------------------------------------------------------
wide["delta_obj_best"] = wide[f"{ALGO_B}_obj_best"] - wide[f"{ALGO_A}_obj_best"]
wide["delta_obj_mean"] = wide[f"{ALGO_B}_obj_mean"] - wide[f"{ALGO_A}_obj_mean"]
wide["delta_runtime"]  = wide[f"{ALGO_B}_runtime_mean"] - wide[f"{ALGO_A}_runtime_mean"]

def verdict(d):
    if pd.isna(d):  return "N/A"
    if d < 0:       return f"{ALGO_B} better"
    if d == 0:      return "Tie"
    return                  f"{ALGO_A} better"

wide["verdict_best"] = wide["delta_obj_best"].map(verdict)
wide["verdict_mean"] = wide["delta_obj_mean"].map(verdict)

# -- Sap xep cot --------------------------------------------------------------
col_order = (
    ["test_id", "mach", "job", "lower_cost", "upper_cost"]
    + [f"{ALGO_A}_obj_best", f"{ALGO_B}_obj_best", "delta_obj_best", "verdict_best"]
    + [f"{ALGO_A}_obj_mean", f"{ALGO_B}_obj_mean", "delta_obj_mean", "verdict_mean"]
    + [f"{ALGO_A}_runtime_mean", f"{ALGO_B}_runtime_mean", "delta_runtime"]
)
wide = wide[[c for c in col_order if c in wide.columns]]
wide["_t"] = wide["test_id"].str.replace("T_", "", regex=False).astype(int)
wide = wide.sort_values("_t").drop(columns=["_t"]).reset_index(drop=True)

# -- Tom tat ra terminal ------------------------------------------------------
both  = wide.dropna(subset=[f"{ALGO_A}_obj_best", f"{ALGO_B}_obj_best"])
total = len(both)

def improv_pct(df, col_winner, col_loser):
    if len(df) == 0:
        return float("nan")
    return ((df[col_loser] - df[col_winner]) / df[col_loser] * 100).mean()

def print_section(metric, col_a, col_b, delta_col, verdict_col):
    counts = both[verdict_col].value_counts()
    tb = both[both[verdict_col] == f"{ALGO_B} better"]
    ib = both[both[verdict_col] == f"{ALGO_A} better"]
    tb_pct = improv_pct(tb, col_b, col_a)
    ib_pct = improv_pct(ib, col_a, col_b)

    print("")
    print(f"  [ {metric} ]")
    for label in [f"{ALGO_B} better", "Tie", f"{ALGO_A} better"]:
        n   = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        print(f"    {label:<22}: {n:>4}  ({pct:5.1f}%)")
    print(f"    Khi {ALGO_B} thang -> tot hon avg : {tb_pct:.2f}%")
    print(f"    Khi {ALGO_A} thang -> tot hon avg : {ib_pct:.2f}%")
    print(f"    Delta avg overall  : {both[delta_col].mean():+.2f}")

SEP = "=" * 60
print(SEP)
print(f"  {ALGO_A}  vs  {ALGO_B}  --  {total} tests co du 2 algo")
print(f"  (tong {len(wide)} tests, {len(wide) - total} chi co 1 algo)")
print(SEP)

print_section("obj_best",
              f"{ALGO_A}_obj_best", f"{ALGO_B}_obj_best",
              "delta_obj_best", "verdict_best")

print_section("obj_avg (mean)",
              f"{ALGO_A}_obj_mean", f"{ALGO_B}_obj_mean",
              "delta_obj_mean", "verdict_mean")

print("")
print(f"  Delta runtime avg : {both['delta_runtime'].mean():+.4f}s")
print(SEP)

# -- Xuat CSV -----------------------------------------------------------------
wide.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nDa xuat: {OUT_CSV}  (rows={len(wide)}, cols={len(wide.columns)})")