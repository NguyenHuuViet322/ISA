"""
So sánh kết quả ISA gốc vs Tabu hybrid + O3 smart.

File 1 (ISA gốc)    : CSV đã tổng hợp, cột: test_id, algo, obj_best, obj_mean, runtime_mean
File 2 (Tabu hybrid): CSV raw, cột: Instance, Run, Objective, Runtime, Iterations

Đầu ra: comparison.csv  (1 hàng / test, wide format dễ nhìn)
"""

from pathlib import Path
import pandas as pd

# ── Đường dẫn file ────────────────────────────────────────────────────────────
ISA_CSV   = Path("summary_obj_by_test.csv")   # file tổng hợp từ summarize.py
TABU_CSV  = Path("tabu_results.csv")           # file raw của Tabu hybrid
OUT_CSV   = Path("comparison.csv")

ALGO_A = "ISA"
ALGO_B = "Tabu+O3"

# ── Đọc ISA ──────────────────────────────────────────────────────────────────
isa = pd.read_csv(ISA_CSV)
# Lọc chỉ lấy file ISA gốc (file đầu tiên theo algo)
first_algo = isa["algo"].iloc[0]
isa = isa[isa["algo"] == first_algo].copy()
isa["test_num"] = isa["test_id"].str.replace("T_", "", regex=False).astype(int)
isa = isa.rename(columns={
    "obj_best":     f"{ALGO_A}_obj_best",
    "obj_mean":     f"{ALGO_A}_obj_mean",
    "runtime_mean": f"{ALGO_A}_runtime_mean",
})

# ── Đọc Tabu hybrid ───────────────────────────────────────────────────────────
tabu_raw = pd.read_csv(TABU_CSV)
tabu_raw.columns = [c.strip() for c in tabu_raw.columns]

tabu = (
    tabu_raw.groupby("Instance", as_index=False)
    .agg(
        obj_best  =("Objective", "min"),
        obj_mean  =("Objective", "mean"),
        runtime_mean=("Runtime", "mean"),
        n_runs    =("Run", "count"),
    )
    .rename(columns={
        "Instance":     "test_num",
        "obj_best":     f"{ALGO_B}_obj_best",
        "obj_mean":     f"{ALGO_B}_obj_mean",
        "runtime_mean": f"{ALGO_B}_runtime_mean",
    })
)

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = pd.merge(
    isa[["test_num", "test_id", "mach", "job", "lower_cost", "upper_cost",
         f"{ALGO_A}_obj_best", f"{ALGO_A}_obj_mean", f"{ALGO_A}_runtime_mean"]],
    tabu[["test_num", f"{ALGO_B}_obj_best", f"{ALGO_B}_obj_mean",
          f"{ALGO_B}_runtime_mean", "n_runs"]],
    on="test_num", how="outer"
).sort_values("test_num").reset_index(drop=True)

# ── Cột so sánh ───────────────────────────────────────────────────────────────
# Δ = Tabu - ISA  (âm = Tabu tốt hơn nếu obj là minimize)
merged["delta_best"] = merged[f"{ALGO_B}_obj_best"] - merged[f"{ALGO_A}_obj_best"]
merged["delta_mean"] = merged[f"{ALGO_B}_obj_mean"] - merged[f"{ALGO_A}_obj_mean"]
merged["delta_runtime"] = merged[f"{ALGO_B}_runtime_mean"] - merged[f"{ALGO_A}_runtime_mean"]

# Kết quả định tính: Tabu tốt hơn / bằng / kém
def verdict(d):
    if pd.isna(d):   return "N/A"
    if d < 0:        return "Tabu better"
    if d == 0:       return "Tie"
    return               "ISA better"

merged["verdict_best"] = merged["delta_best"].map(verdict)
merged["verdict_mean"] = merged["delta_mean"].map(verdict)

# ── In tóm tắt ra màn hình ────────────────────────────────────────────────────
total   = merged["verdict_best"].notna().sum()
counts  = merged["verdict_best"].value_counts()

print("=" * 55)
print(f"  So sánh {ALGO_A} vs {ALGO_B}  ({total} tests)")
print("=" * 55)
for label in ["Tabu better", "Tie", "ISA better", "N/A"]:
    n = counts.get(label, 0)
    bar = "█" * n
    print(f"  {label:<14}: {n:>4}  {bar}")
print()
print(f"  Δ obj_best  trung bình : {merged['delta_best'].mean():+.2f}")
print(f"  Δ obj_mean  trung bình : {merged['delta_mean'].mean():+.2f}")
print(f"  Δ runtime   trung bình : {merged['delta_runtime'].mean():+.4f}s")
print("=" * 55)

# ── Xuất CSV ──────────────────────────────────────────────────────────────────
merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nĐã xuất: {OUT_CSV}  (rows={len(merged)}, cols={len(merged.columns)})")