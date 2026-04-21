import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot ISA vs Hybrid Tabu")
parser.add_argument("--isa",  default="batch_results_ISA.csv")
parser.add_argument("--tabu", default="tabu_results.csv")
parser.add_argument("--out",  default="plots")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
isa  = pd.read_csv(args.isa)
tabu = pd.read_csv(args.tabu)

# ─────────────────────────────────────────────────────────────
# Aggregate theo Instance
# ─────────────────────────────────────────────────────────────
def summarize(df):
    return df.groupby("Instance").agg(
        obj_best=("Objective", "min"),
        obj_avg=("Objective", "mean"),
        runtime_avg=("Runtime", "mean"),
        iter_avg=("Iterations", "mean")
    ).reset_index()

isa_sum  = summarize(isa)
tabu_sum = summarize(tabu)

# Merge
df = isa_sum.merge(tabu_sum, on="Instance", suffixes=("_ISA", "_TABU"))

# Tạo folder output
os.makedirs(args.out, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────
def save_plot(filename):
    plt.savefig(os.path.join(args.out, filename), dpi=300, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────
# 1. Average Objective
# ─────────────────────────────────────────────────────────────
plt.figure()
plt.plot(df["Instance"], df["obj_avg_ISA"], marker='o', label="ISA")
plt.plot(df["Instance"], df["obj_avg_TABU"], marker='s', label="Hybrid Tabu")
plt.xlabel("Instance")
plt.ylabel("Average Objective")
plt.title("Average Objective Comparison")
plt.legend()
plt.grid()
save_plot("obj_avg.png")

# ─────────────────────────────────────────────────────────────
# 2. Best Objective
# ─────────────────────────────────────────────────────────────
plt.figure()
plt.plot(df["Instance"], df["obj_best_ISA"], marker='o', label="ISA")
plt.plot(df["Instance"], df["obj_best_TABU"], marker='s', label="Hybrid Tabu")
plt.xlabel("Instance")
plt.ylabel("Best Objective")
plt.title("Best Objective Comparison")
plt.legend()
plt.grid()
save_plot("obj_best.png")

# ─────────────────────────────────────────────────────────────
# 3. Runtime
# ─────────────────────────────────────────────────────────────
plt.figure()
plt.plot(df["Instance"], df["runtime_avg_ISA"], marker='o', label="ISA")
plt.plot(df["Instance"], df["runtime_avg_TABU"], marker='s', label="Hybrid Tabu")
plt.xlabel("Instance")
plt.ylabel("Average Runtime (s)")
plt.title("Runtime Comparison")
plt.legend()
plt.grid()
save_plot("runtime.png")

# ─────────────────────────────────────────────────────────────
# 4. Iterations
# ─────────────────────────────────────────────────────────────
plt.figure()
plt.plot(df["Instance"], df["iter_avg_ISA"], marker='o', label="ISA")
plt.plot(df["Instance"], df["iter_avg_TABU"], marker='s', label="Hybrid Tabu")
plt.xlabel("Instance")
plt.ylabel("Average Iterations")
plt.title("Iterations Comparison")
plt.legend()
plt.grid()
save_plot("iterations.png")

# ─────────────────────────────────────────────────────────────
# 5. Boxplot Objective (quan trọng cho paper)
# ─────────────────────────────────────────────────────────────
plt.figure()
plt.boxplot(
    [isa["Objective"], tabu["Objective"]],
    labels=["ISA", "Hybrid Tabu"]
)
plt.ylabel("Objective")
plt.title("Objective Distribution")
save_plot("boxplot_objective.png")

# ─────────────────────────────────────────────────────────────
# 6. GAP (%) (ISA vs Tabu)
# ─────────────────────────────────────────────────────────────
df["gap_%"] = (df["obj_avg_ISA"] - df["obj_avg_TABU"]) / df["obj_avg_TABU"] * 100

plt.figure()
plt.plot(df["Instance"], df["gap_%"], marker='o')
plt.axhline(0)
plt.xlabel("Instance")
plt.ylabel("Gap (%)")
plt.title("Performance Gap (ISA vs Hybrid Tabu)")
plt.grid()
save_plot("gap.png")

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print(f"Saved all plots to folder: {args.out}")