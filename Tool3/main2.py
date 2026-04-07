import pandas as pd
from pathlib import Path

# ====== INPUTS (bạn sửa đường dẫn cho đúng) ======
ORIGINAL_CSV = Path("summary_mean_obj_by_test.csv")   # file output từ script trước
BATCH_RESULT = Path("output.csv")              # file ISA (ảnh bạn gửi)
OUT_CSV      = Path("merged_original_vs_isa_new.csv")

def read_batch_result(path: Path) -> pd.DataFrame:
    """
    Batch Result có thể là CSV chuẩn, hoặc file text có dấu phẩy.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Chuẩn hoá tên cột
    # Expected: Instance, Run, Objective, Runtime
    rename_map = {
        "instance": "Instance",
        "run": "Run",
        "objective": "Objective",
        "runtime": "Runtime",
    }
    df = df.rename(columns={c: rename_map.get(c.lower(), c) for c in df.columns})

    for c in ["Instance", "Objective"]:
        if c not in df.columns:
            raise ValueError(f"Batch result thiếu cột '{c}'. Cột hiện có: {list(df.columns)}")

    # Convert numeric
    df["Instance"] = pd.to_numeric(df["Instance"], errors="coerce").astype("Int64")
    df["Objective"] = pd.to_numeric(df["Objective"], errors="coerce")

    # Bỏ dòng lỗi
    df = df.dropna(subset=["Instance", "Objective"])

    return df

def main():
    # --- Load original summary ---
    orig = pd.read_csv(ORIGINAL_CSV)
    orig.columns = [c.strip() for c in orig.columns]

    if "test_id" not in orig.columns:
        raise ValueError("Original summary phải có cột 'test_id' (ví dụ T_1).")

    # test_id -> i
    orig["Instance"] = (
        orig["test_id"]
        .astype(str)
        .str.replace("T_", "", regex=False)
        .astype(int)
    )

    # rename mean_obj -> obj_original
    if "mean_obj" not in orig.columns:
        raise ValueError("Original summary phải có cột 'mean_obj'.")
    orig = orig.rename(columns={"mean_obj": "obj_original"})

    # --- Load ISA batch result ---
    isa = read_batch_result(BATCH_RESULT)

    # Nếu có nhiều Run cho cùng Instance: lấy trung bình Objective, Runtime
    group_cols = ["Instance"]
    agg = {"Objective": "mean"}
    if "Runtime" in isa.columns:
        isa["Runtime"] = pd.to_numeric(isa["Runtime"], errors="coerce")
        agg["Runtime"] = "mean"

    isa_g = isa.groupby(group_cols, as_index=False).agg(agg)
    isa_g = isa_g.rename(columns={"Objective": "obj_ISA", "Runtime": "runtime_ISA"})

    # --- Merge ---
    merged = orig.merge(isa_g, on="Instance", how="outer")

    # Thêm vài cột tiện so sánh
    merged["gap_ISA_minus_original"] = merged["obj_ISA"] - merged["obj_original"]
    merged["ratio_ISA_over_original"] = merged["obj_ISA"] / merged["obj_original"]

    # Sắp xếp theo Instance
    merged = merged.sort_values("Instance")

    # Đưa test_id lại từ Instance nếu bị NaN phía ISA-only
    merged["test_id"] = merged["test_id"].fillna(merged["Instance"].map(lambda x: f"T_{int(x)}" if pd.notna(x) else None))

    # Reorder columns cho dễ nhìn
    cols_prefer = [
        "Instance", "test_id",
        "obj_original", "obj_ISA",
        "gap_ISA_minus_original", "ratio_ISA_over_original",
        "runtime_ISA",
        "file", "n_rows", "lower_cost", "upper_cost",
    ]
    cols_final = [c for c in cols_prefer if c in merged.columns] + [c for c in merged.columns if c not in cols_prefer]
    merged = merged[cols_final]

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done -> {OUT_CSV} (rows={len(merged)})")

    # Báo nhanh các instance thiếu bên nào
    missing_orig = merged[merged["obj_original"].isna()]["Instance"].dropna().tolist()
    missing_isa  = merged[merged["obj_ISA"].isna()]["Instance"].dropna().tolist()
    if missing_orig:
        print(f"[WARN] Có {len(missing_orig)} instance chỉ có ISA, thiếu original. Ví dụ: {missing_orig[:10]}")
    if missing_isa:
        print(f"[WARN] Có {len(missing_isa)} instance chỉ có original, thiếu ISA. Ví dụ: {missing_isa[:10]}")

if __name__ == "__main__":
    main()