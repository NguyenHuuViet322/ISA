import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUT_CSV = "summary_mean_obj_by_test.csv"

def parse_test_range_from_filename(name: str):
    m = re.search(r"[Ii]_(\d+)_(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def read_table_any(path: Path) -> pd.DataFrame:
    head = path.read_bytes()[:128]

    # TSV text giả .xls
    if b"\t" in head and (b"obj\t" in head or b"obj\titer" in head):
        return pd.read_csv(path, sep="\t", engine="python")
    # CSV text giả .xls
    if b"," in head and (b"obj," in head or b"obj,iter" in head):
        return pd.read_csv(path, sep=",", engine="python")

    # Excel thật
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    else:
        # pip install xlrd==2.0.1 (chỉ khi có xls binary thật)
        return pd.read_excel(path, engine="xlrd")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip + lower
    df.columns = [str(c).strip() for c in df.columns]

    # bỏ các cột Unnamed
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]

    # map các biến thể tên cột về chuẩn
    col_map = {
        "lower_cos": "lower_cost",
        "lower_cost": "lower_cost",
        "upper_csot": "upper_cost",
        "upper_cost": "upper_cost",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    return df

def process_file(path: Path) -> pd.DataFrame:
    df = read_table_any(path)
    df = normalize_columns(df)

    required = ["obj", "lower_cost", "upper_cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing}. Cột hiện có: {list(df.columns)}")

    # --- thêm 3 cột mới ---
    extra_cols = ["mach", "job", "runtime"]
    missing_extra = [c for c in extra_cols if c not in df.columns]
    if missing_extra:
        raise ValueError(f"Thiếu cột {missing_extra} trong file {path.name}")

    start_test, end_test = parse_test_range_from_filename(path.stem)
    if start_test is None:
        raise ValueError(f"Không parse được range test từ tên file: {path.name}")

    # mỗi lần (lower_cost, upper_cost) đổi => test mới
    cost = df[["lower_cost", "upper_cost"]].copy()
    changed = (cost != cost.shift(1)).any(axis=1)
    changed.iloc[0] = True
    seg_id = changed.cumsum()

    df["_test_num"] = start_test + (seg_id - 1)

    out = (
        df.groupby("_test_num", as_index=False)
          .agg(
              mean_obj=("obj", "mean"),
              n_rows=("obj", "size"),
              lower_cost=("lower_cost", "first"),
              upper_cost=("upper_cost", "first"),
              mach=("mach", "first"),
              job=("job", "first"),
              mean_runtime=("runtime", "mean"),   # lấy trung bình runtime
          )
    )

    out.insert(0, "file", path.name)
    out.insert(1, "test_id", out["_test_num"].map(lambda x: f"T_{x}"))
    out = out.drop(columns=["_test_num"])

    if end_test is not None:
        max_test = int(out["test_id"].str.replace("T_", "", regex=False).max())
        if max_test > end_test:
            print(f"[WARN] {path.name}: test tới T_{max_test} > T_{end_test} (check dữ liệu)")

    return out

def main():
    files = sorted(
        list(DATA_DIR.glob("*.xls")) + list(DATA_DIR.glob("*.xlsx")),
        key=lambda p: (parse_test_range_from_filename(p.stem)[0] or 10**18, p.name)
    )
    if not files:
        raise SystemExit(f"Không tìm thấy file trong {DATA_DIR.resolve()}")

    all_rows = []
    for p in files:
        try:
            all_rows.append(process_file(p))
            print(f"OK: {p.name}")
        except Exception as e:
            print(f"FAIL: {p.name} -> {e}")

    if not all_rows:
        raise SystemExit("Không có file nào xử lý thành công.")

    result = pd.concat(all_rows, ignore_index=True)
    result["_t"] = result["test_id"].str.replace("T_", "", regex=False).astype(int)
    result = result.sort_values(["_t", "file"]).drop(columns=["_t"])

    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nĐã xuất: {OUT_CSV} (rows={len(result)})")

if __name__ == "__main__":
    main()