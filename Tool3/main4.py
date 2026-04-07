import csv
from collections import defaultdict

input_file = "batch_results6.csv"
output_file = "output.csv"

# Lưu dữ liệu theo instance
data = defaultdict(list)

# Đọc CSV
with open(input_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        instance = int(row["Instance"])
        run = int(row["Run"])
        obj = float(row["Objective"])
        runtime = float(row["Runtime"])
        data[instance].append((run, obj, runtime))

# Ghi CSV output
with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Instance", "Run", "Objective", "Runtime"])

    for instance in sorted(data.keys()):
        runs = data[instance]

        # Sắp xếp theo objective tăng dần
        runs_sorted = sorted(runs, key=lambda x: x[1])

        # Lấy 8 cái tốt nhất
        best8 = runs_sorted[:8]

        # Tính trung bình
        avg_obj = sum(x[1] for x in best8) / len(best8)
        avg_runtime = sum(x[2] for x in best8) / len(best8)

        # Xuất ra 1 dòng / instance
        writer.writerow([instance, 1, round(avg_obj, 4), round(avg_runtime, 6)])

print("Done!")