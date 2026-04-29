#!/usr/bin/env bash

set -euo pipefail

# Generates live telemetry data for training by running the baseline scenario multiple times.
# We use baseline because it does not adapt, meaning the model learns pure, unadulterated 
# network impact across all performance zones.

RUNS="${1:-3}"
DURATION="${2:-660}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=========================================================="
echo " Starting LIVE Dataset Generation"
echo " Runs: ${RUNS}"
echo " Duration per run: ${DURATION}s"
echo "=========================================================="

# Create a special directory to store these training runs
TRAIN_DIR="${PROJECT_DIR}/outputs/training_datasets"
mkdir -p "${TRAIN_DIR}"

for i in $(seq 1 "$RUNS"); do
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " Generating Training Run $i of $RUNS"
    echo "──────────────────────────────────────────────────────────"
    
    # We run the baseline script directly.
    # It will output to outputs/evaluation/baseline_natural_TIMESTAMP
    bash "${PROJECT_DIR}/scripts/evaluation/run_natural_stress_baseline.sh" "$DURATION"
    
    # Rest to let sockets and kernel buffers completely clear
    echo "[!] Run $i complete. Cooling down for 10 seconds..."
    sleep 10
done

echo ""
echo "=========================================================="
echo " Dataset generation complete. Data is in outputs/evaluation"
echo " Run retrain_on_live_data.py to update the XGBoost model."
echo "=========================================================="
