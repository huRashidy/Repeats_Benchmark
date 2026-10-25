#!/usr/bin/env bash
set -e

PYTHON="/home/hussin/miniconda3/envs/benchmark_env/bin/python"

echo "==========================================================="
echo "Starting Full Carbon-X Benchmark on 2 x RTX 2080 Ti GPUs"
echo "==========================================================="

echo ""
echo "[Step 1/2] Running Repeat Class Prediction (16 classes, 2000 samples)..."
rm -f results/predictions_class.csv results/metrics_class.json results/confusion_matrix_class.png
$PYTHON src/predict.py --task class

echo ""
echo "[Step 2/2] Running Repeat Family Prediction (49 families, 2000 samples)..."
rm -f results/predictions_family.csv results/metrics_family.json results/confusion_matrix_family.png
$PYTHON src/predict.py --task family

echo ""
echo "==========================================================="
echo "Full Benchmark Run Complete!"
echo "Results available in results/"
echo "==========================================================="
