#!/usr/bin/env bash
set -e

PYTHON="/home/hussin/miniconda3/envs/benchmark_env/bin/python"

echo "==========================================================="
echo "Starting Full Carbon-X Benchmark on 4 Datasets"
echo "Hardware: 2 x NVIDIA RTX 2080 Ti (8-bit Quantized)"
echo "==========================================================="

echo ""
echo "[1/4] Running Repeat Class (15 classes, N=2000)..."
rm -f results/predictions_class.csv results/metrics_class.json results/confusion_matrix_class.png
$PYTHON src/predict.py --task class

echo ""
echo "[2/4] Running Repeat Family (48 families, N=2000)..."
rm -f results/predictions_family.csv results/metrics_family.json results/confusion_matrix_family.png
$PYTHON src/predict.py --task family

echo ""
echo "[3/4] Running Repeat Class [Min 100bp] (14 classes, N=2000)..."
rm -f results/predictions_class_min100.csv results/metrics_class_min100.json results/confusion_matrix_class_min100.png
$PYTHON src/predict.py --task class_min100

echo ""
echo "[4/4] Running Repeat Family [Min 100bp] (47 families, N=2000)..."
rm -f results/predictions_family_min100.csv results/metrics_family_min100.json results/confusion_matrix_family_min100.png
$PYTHON src/predict.py --task family_min100

echo ""
echo "==========================================================="
echo "All 4 Benchmark Runs Complete!"
echo "Results available in results/"
echo "==========================================================="
