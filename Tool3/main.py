import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUT_CSV  = "summary_obj_by_test.csv"

# ── Các cột thống kê giữ lại cho mỗi file × test ────────────────────────────
# Thay đổi danh sách này nếu muốn thêm/bớt cột
KEEP_STATS = [
    "obj_best",
    "obj_mean",
    "runtime_mean",
]

# ── Cột chung (giống nhau giữa các file, chỉ lấy 1 lần) ─────────────────────
SHARED_COLS = ["mach", "job", "lower_cost", "upper_cost"]

# ────────────────────────────────────────────────────────────────────────────

def parse_test_range_from_filename(name: str):
    m = re.search(r"[Ii]_(\d+)_(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def read_table_any(path: Path) -> pd.DataFrame:
    head = path.read_bytes()[:128]
    if b"\t" in head and (b"obj\t" in head or b"obj\titer" in head):
        return pd.read_csv(path, sep="\t", engine="python")
    if b"," in head and (b"obj," in head or b"obj,iter" in head):
        return pd.read_csv(path, sep=",", engine="python")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_excel(path, engine="xlrd")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]
    col_map = {
        "lower_cos":  "lower_cost",
        "lower_cost": "lower_cost",
        "upper_csot": "upper_cost",
        "upper_cost": "upper_cost",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    return df

def agg_group(g: pd.DataFrame) -> pd.Series:
    obj = g["obj"]
    rt  = g["runtime"]
    best_idx = obj.idxmin()
    return pd.Series({
        "obj_best":            obj.min(),
        "obj_mean":            obj.mean(),
        "obj_std":             obj.std(ddof=1) if len(obj) > 1 else 0.0,
        "runtime_mean":        rt.mean(),
        "runtime_at_best_obj": g.loc[best_idx, "runtime"],
        "n_runs":              len(obj),
        "n_best_hit":          int((obj == obj.min()).sum()),
        # shared
        "lower_cost":          g["lower_cost"].iloc[0],
        "upper_cost":          g["upper_cost"].iloc[0],
        "mach":                g["mach"].iloc[0],
        "job":                 g["job"].iloc[0],
    })

def process_file(path: Path) -> pd.DataFrame:
    df = read_table_any(path)
    df = normalize_columns(df)

    required = ["obj", "lower_cost", "upper_cost", "mach", "job", "runtime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing}. Cột hiện có: {list(df.columns)}")

    start_test, end_test = parse_test_range_from_filename(path.stem)
    if start_test is None:
        raise ValueError(f"Không parse được range test từ tên file: {path.name}")

    cost    = df[["lower_cost", "upper_cost"]].copy()
    changed = (cost != cost.shift(1)).any(axis=1)
    changed.iloc[0] = True
    df["_test_num"] = start_test + (changed.cumsum() - 1)

    out = (
        df.groupby("_test_num", as_index=False)
          .apply(agg_group)
          .reset_index(drop=True)
    )
    out["test_id"] = out["_test_num"].map(lambda x: f"T_{x}")
    out = out.drop(columns=["_test_num"])

    if end_test is not None:
        max_t = out["test_id"].str.replace("T_", "", regex=False).astype(int).max()
        if max_t > end_test:
            print(f"[WARN] {path.name}: test tới T_{max_t} > T_{end_test}")

    return out

def make_short_label(path: Path, existing: set) -> str:
    """Rút gọn tên file thành label ngắn dùng làm prefix cột, đảm bảo unique."""
    stem = path.stem
    stem = re.sub(r"[_\-][Ii]_\d+_\d+$", "", stem)   # bỏ _I_x_y ở cuối
    label = stem
    if label in existing:
        label = f"{stem}_{path.stem[-4:]}"
    return label

def main():
    files = sorted(
        list(DATA_DIR.glob("*.xls")) + list(DATA_DIR.glob("*.xlsx")),
        key=lambda p: (parse_test_range_from_filename(p.stem)[0] or 10**18, p.name)
    )
    if not files:
        raise SystemExit(f"Không tìm thấy file trong {DATA_DIR.resolve()}")

    frames: dict[str, pd.DataFrame] = {}
    used_labels: set = set()

    for p in files:
        try:
            df = process_file(p)
            label = make_short_label(p, used_labels)
            used_labels.add(label)
            frames[label] = df.set_index("test_id")
            print(f"OK: {p.name}  →  label='{label}'")
        except Exception as e:
            print(f"FAIL: {p.name} -> {e}")

    if not frames:
        raise SystemExit("Không có file nào xử lý thành công.")

    # ── Long format: mỗi file × test = 1 hàng ───────────────────────────────
    rows = []
    for label, df in frames.items():
        sub = df[SHARED_COLS + KEEP_STATS].copy()
        sub.insert(0, "algo", label)
        rows.append(sub)

    result = pd.concat(rows, axis=0).reset_index()   # test_id trở lại thành cột

    # Sắp xếp theo số test rồi theo algo
    result["_t"] = result["test_id"].str.replace("T_", "", regex=False).astype(int)
    result = result.sort_values(["_t", "algo"]).drop(columns=["_t"]).reset_index(drop=True)

    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nĐã xuất: {OUT_CSV}  (rows={len(result)}, cols={len(result.columns)})")
    print("Cột:", list(result.columns))

if __name__ == "__main__":
    main()