"""
process_tabu.py
───────────────
Đọc file kết quả Tabu hybrid raw (cột: Instance, Run, Objective, Runtime, Iterations),
tổng hợp theo Instance, rồi join mach/job/lower_cost/upper_cost từ file ISA summary
để ra cùng format với summary_obj_by_test.csv.

Đầu ra: tabu_summary.csv
Cột   : test_id, algo, mach, job, lower_cost, upper_cost, obj_best, obj_mean, runtime_mean
"""

from pathlib import Path
import pandas as pd

# ── Đường dẫn ────────────────────────────────────────────────────────────────
TABU_RAW  = Path("tabu_results.csv")       # file raw Tabu hybrid
ISA_SUMM  = Path("summary_obj_by_test.csv") # file tổng hợp ISA (để lấy shared cols)
OUT_CSV   = Path("tabu_summary.csv")

ALGO_LABEL = "Tabu+O3"

# ── Đọc file Tabu raw ─────────────────────────────────────────────────────────
df = pd.read_csv(TABU_RAW)
df.columns = [c.strip() for c in df.columns]

# Rename về tên chuẩn
df = df.rename(columns={
    "Instance":   "test_num",
    "Run":        "run",
    "Objective":  "obj",
    "Runtime":    "runtime",
    "Iterations": "iterations",
})

# ── Tổng hợp theo test_num ────────────────────────────────────────────────────
def agg_group(g):
    obj = g["obj"]
    rt  = g["runtime"]
    return pd.Series({
        "obj_best":     obj.min(),
        "obj_mean":     obj.mean(),
        "runtime_mean": rt.mean(),
    })

tabu = (
    df.groupby("test_num", as_index=False)
      .apply(agg_group)
      .reset_index(drop=True)
)
tabu["test_id"] = tabu["test_num"].map(lambda x: f"T_{x}")

# ── Lấy shared cols từ ISA summary ───────────────────────────────────────────
isa = pd.read_csv(ISA_SUMM)

# Lấy 1 dòng đại diện mỗi test (shared cols giống nhau giữa các algo)
shared = (
    isa[["test_id", "mach", "job", "lower_cost", "upper_cost"]]
    .drop_duplicates("test_id")
)
shared["test_num"] = shared["test_id"].str.replace("T_", "", regex=False).astype(int)

# ── Join ──────────────────────────────────────────────────────────────────────
tabu = tabu.merge(shared[["test_num", "mach", "job", "lower_cost", "upper_cost"]],
                  on="test_num", how="left")

# Cảnh báo nếu test nào không có shared info
missing = tabu[tabu["mach"].isna()]["test_num"].tolist()
if missing:
    print(f"[WARN] {len(missing)} test không tìm thấy shared info từ ISA: {missing[:10]}...")

# ── Sắp xếp và xuất ──────────────────────────────────────────────────────────
tabu = tabu.sort_values("test_num").reset_index(drop=True)

result = tabu[[
    "test_id", "mach", "job", "lower_cost", "upper_cost",
    "obj_best", "obj_mean", "runtime_mean"
]].copy()
result.insert(1, "algo", ALGO_LABEL)

result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"Đã xuất: {OUT_CSV}  (rows={len(result)})")
print("Cột:", list(result.columns))