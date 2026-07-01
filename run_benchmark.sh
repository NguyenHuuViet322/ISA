#!/bin/bash
set -e

SRC_DIR="."
BIN_NAME="solver"
RUNS=${1:-10}     # số run/instance, mặc định 10: ./run_benchmark.sh 20

echo "=== Building ==="
g++ -O3 -std=c++17 -o "$SRC_DIR/$BIN_NAME" \
    "$SRC_DIR/main.cpp" \
    -I"$SRC_DIR"

echo ""
echo "=== Running benchmark (--batch, T_1..T_2160) ==="
echo "  Small  (n<=12, m<=3)  : 10s / run"
echo "  Medium (n<=50, m<=10) : 100s / run"
echo "  Large  (còn lại)      : 300s / run"
echo "  Runs/instance         : $RUNS"
echo ""

"$SRC_DIR/$BIN_NAME" --batch "$RUNS"

echo ""
echo "=== Xong. Kết quả tại batch_results6.csv ==="