import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load CSV file ---
df = pd.read_csv("merged_original_vs_isa.csv")  # sửa tên file

# --- 2. Remove rows where obj_original is missing ---
df_clean = df.dropna(subset=["obj_original"]).copy()

# --- 3. Compute relative gap (negative = improvement) ---
df_clean["relative_gap"] = (
    (df_clean["obj_ISA"] - df_clean["obj_original"])
    / df_clean["obj_original"] * 100
)

# --- 4. Check available machine sizes ---
machine_sizes = sorted(df_clean["mach"].unique())
print("Machine sizes found:", machine_sizes)

for m in machine_sizes:
    subset = df_clean[df_clean["mach"] == m]["relative_gap"]
    print(f"Machine {m}")
    print("  min:", subset.min())
    print("  max:", subset.max())
    print("  mean:", subset.mean())
    print("  unique values:", subset.nunique())
    print()

# --- 5. Prepare data grouped by machine size ---
data_to_plot = [
    df_clean[df_clean["mach"] == m]["relative_gap"]
    for m in machine_sizes
]

# --- 6. Plot ---
plt.figure()
plt.boxplot(data_to_plot)
plt.xticks(range(1, len(machine_sizes) + 1), machine_sizes)
plt.axhline(0)
plt.xlabel("Number of Machines")
plt.ylabel("Relative Gap (%)")
plt.title("Distribution of Relative Gap by Machine Size")
plt.show()