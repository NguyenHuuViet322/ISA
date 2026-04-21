import pandas as pd

def summarize(df):
    grouped = df.groupby(["Folder", "Instance"])

    summary = pd.DataFrame({
        "obj_best": grouped["Objective"].min(),
        "obj_avg": grouped["Objective"].mean(),
        "obj_std": grouped["Objective"].std(),
        "iter_avg": grouped["Iterations"].mean(),
        "runtime_avg": grouped["Runtime"].mean(),
        "n_runs": grouped["Run"].count()
    }).reset_index()

    return summary


def compare(df1, df2):
    t1 = summarize(df1)
    t2 = summarize(df2)

    merged = pd.merge(
        t1, t2,
        on=["Folder", "Instance"],
        suffixes=("_T1", "_T2")
    )

    # ===== BEST =====
    merged["delta_best"] = merged["obj_best_T2"] - merged["obj_best_T1"]
    merged["delta_best_pct"] = merged["delta_best"] / merged["obj_best_T1"] * 100

    def verdict_best(row):
        if row["obj_best_T1"] < row["obj_best_T2"]:
            return "TabuO3"
        elif row["obj_best_T1"] > row["obj_best_T2"]:
            return "TabuO3Smart"
        else:
            return "Tie"

    merged["verdict_best"] = merged.apply(verdict_best, axis=1)

    # ===== AVG =====
    merged["delta_avg"] = merged["obj_avg_T2"] - merged["obj_avg_T1"]
    merged["delta_avg_pct"] = merged["delta_avg"] / merged["obj_avg_T1"] * 100

    def verdict_avg(row):
        if row["obj_avg_T1"] < row["obj_avg_T2"]:
            return "TabuO3"
        elif row["obj_avg_T1"] > row["obj_avg_T2"]:
            return "TabuO3Smart"
        else:
            return "Tie"

    merged["verdict_avg"] = merged.apply(verdict_avg, axis=1)

    # ===== OUTPUT =====
    result = pd.DataFrame({
        "Folder": merged["Folder"],
        "Instance": merged["Instance"],

        "TabuO3_obj_best": merged["obj_best_T1"],
        "TabuO3Smart_obj_best": merged["obj_best_T2"],
        "delta_best": merged["delta_best"],
        "delta_best_pct": merged["delta_best_pct"],
        "verdict_best": merged["verdict_best"],

        "TabuO3_obj_avg": merged["obj_avg_T1"],
        "TabuO3Smart_obj_avg": merged["obj_avg_T2"],
        "delta_avg": merged["delta_avg"],
        "delta_avg_pct": merged["delta_avg_pct"],
        "verdict_avg": merged["verdict_avg"],

        "TabuO3_obj_std": merged["obj_std_T1"],
        "TabuO3Smart_obj_std": merged["obj_std_T2"],

        "TabuO3_iter_avg": merged["iter_avg_T1"],
        "TabuO3Smart_iter_avg": merged["iter_avg_T2"],

        "TabuO3_runtime_avg": merged["runtime_avg_T1"],
        "TabuO3Smart_runtime_avg": merged["runtime_avg_T2"],

        "TabuO3_n_runs": merged["n_runs_T1"],
        "TabuO3Smart_n_runs": merged["n_runs_T2"]
    })

    return result


# ===== MAIN =====
df1 = pd.read_csv("bigdata_tabuO3.csv")
df2 = pd.read_csv("bigdata_tabuO3smart.csv")

result = compare(df1, df2)

result.to_csv("comparison.csv", index=False)

print("Done!")