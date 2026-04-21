"""
Tạo bảng tổng hợp so sánh ISA (Standard) vs Hybrid Tabu.
Metrics: AVG_DIFF% và BEST_DIFF%  (dương = Hybrid Tabu tốt hơn ISA)

Cấu trúc CSV:
  ISA file:   Folder, Instance, Mode, Run, Objective, Runtime, Iterations  (lọc Mode=STANDARD)
  Tabu file:  Folder, Instance, Mode, Run, Objective, Runtime, Iterations

Chạy:
    python summary_table.py --isa isa_results.csv --tabu tabu_results.csv
    python summary_table.py --isa isa_results.csv --tabu tabu_results.csv --out results_table.csv

Yêu cầu: pandas
"""

import argparse
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--isa",  default="o3_results.csv")
parser.add_argument("--tabu", default="tabu_results_big.csv")
parser.add_argument("--out",  default="results_table.csv", help="Output CSV path")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
isa_raw  = pd.read_csv(args.isa)
tabu_raw = pd.read_csv(args.tabu)
isa_raw.columns  = isa_raw.columns.str.strip()
tabu_raw.columns = tabu_raw.columns.str.strip()

# Chỉ lấy STANDARD từ ISA
isa_raw = isa_raw[isa_raw["Mode"].str.strip().str.upper() == "STANDARD"].copy()

# ── Aggregate per (Folder, Instance) ─────────────────────────────────────────
def agg(df):
    return (df.groupby(["Folder", "Instance"])
              .agg(
                  avg  =("Objective", "mean"),
                  best =("Objective", "min"),
              )
              .reset_index())

isa_agg  = agg(isa_raw).rename(columns={"avg": "ISA_avg", "best": "ISA_best"})
tabu_agg = agg(tabu_raw).rename(columns={"avg": "Tabu_avg", "best": "Tabu_best"})

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = pd.merge(isa_agg, tabu_agg, on=["Folder", "Instance"])

# ── Compute diff (positive = Hybrid Tabu better = lower objective) ────────────
# avg_diff%  = (ISA_avg  - Tabu_avg)  / ISA_avg  * 100
# best_diff% = (ISA_best - Tabu_best) / ISA_best * 100
merged["AVG_DIFF%"]  = (merged["ISA_avg"]  - merged["Tabu_avg"])  / merged["ISA_avg"]  * 100
merged["BEST_DIFF%"] = (merged["ISA_best"] - merged["Tabu_best"]) / merged["ISA_best"] * 100

# Round
for col in ["ISA_avg", "ISA_best", "Tabu_avg", "Tabu_best"]:
    merged[col] = merged[col].round(2)
merged["AVG_DIFF%"]  = merged["AVG_DIFF%"].round(3)
merged["BEST_DIFF%"] = merged["BEST_DIFF%"].round(3)

# ── Add verdict column ────────────────────────────────────────────────────────
def verdict(row):
    avg_v  = "Tabu" if row["AVG_DIFF%"]  > 0 else ("ISA" if row["AVG_DIFF%"]  < 0 else "Tie")
    best_v = "Tabu" if row["BEST_DIFF%"] > 0 else ("ISA" if row["BEST_DIFF%"] < 0 else "Tie")
    return avg_v, best_v

merged[["Verdict_avg", "Verdict_best"]] = merged.apply(
    lambda r: pd.Series(verdict(r)), axis=1)

# ── T-AVG per Folder ─────────────────────────────────────────────────────────
tavg_rows = []
for folder, grp in merged.groupby("Folder"):
    tavg_rows.append({
        "Folder":      folder,
        "Instance":    "T-AVG",
        "ISA_avg":     "",
        "ISA_best":    "",
        "Tabu_avg":    "",
        "Tabu_best":   "",
        "AVG_DIFF%":   round(grp["AVG_DIFF%"].mean(),  3),
        "BEST_DIFF%":  round(grp["BEST_DIFF%"].mean(), 3),
        "Verdict_avg":  "",
        "Verdict_best": "",
    })

tavg_df = pd.DataFrame(tavg_rows)

# ── Interleave T-AVG rows after each folder block ────────────────────────────
final_rows = []
for folder, grp in merged.groupby("Folder"):
    final_rows.append(grp)
    final_rows.append(tavg_df[tavg_df["Folder"] == folder])

final = pd.concat(final_rows, ignore_index=True)

# ── Column order ──────────────────────────────────────────────────────────────
final = final[["Folder", "Instance",
               "ISA_avg", "Tabu_avg", "AVG_DIFF%", "Verdict_avg",
               "ISA_best", "Tabu_best", "BEST_DIFF%", "Verdict_best"]]

# ── Print to console ──────────────────────────────────────────────────────────
pd.set_option("display.max_rows", 999)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 140)
pd.set_option("display.colheader_justify", "center")
print(final.to_string(index=False))

# ── Save CSV ──────────────────────────────────────────────────────────────────
final.to_csv(args.out, index=False)
print(f"\nSaved: {args.out}")