#!/usr/bin/env bash
# Download canonical Iris and Optical Handwritten Digits datasets from the UCI Machine Learning Repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "============================================================"
echo "   Downloading Toy Datasets for Neural Network Engine"
echo "============================================================"
echo "Target directory: ${DATA_DIR}"
echo ""

# 1. Fisher's Iris Dataset (150 samples, 4 features, 3 classes)
IRIS_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
echo "[1/3] Downloading Fisher's Iris Dataset..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "${IRIS_URL}" -o "${DATA_DIR}/iris.data"
elif command -v wget >/dev/null 2>&1; then
    wget -q "${IRIS_URL}" -O "${DATA_DIR}/iris.data"
else
    echo "Error: Neither curl nor wget found on system." >&2
    exit 1
fi
echo "  -> Saved to ${DATA_DIR}/iris.data ($(wc -l < "${DATA_DIR}/iris.data" | tr -d ' ') lines)"

# 2. Optical Recognition of Handwritten Digits - Training Set (3,823 samples, 8x8 pixels, 10 classes)
OPTDIGITS_TRA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
echo "[2/3] Downloading 8x8 Optical Digits (Training Set)..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "${OPTDIGITS_TRA_URL}" -o "${DATA_DIR}/optdigits.tra"
elif command -v wget >/dev/null 2>&1; then
    wget -q "${OPTDIGITS_TRA_URL}" -O "${DATA_DIR}/optdigits.tra"
fi
echo "  -> Saved to ${DATA_DIR}/optdigits.tra ($(wc -l < "${DATA_DIR}/optdigits.tra" | tr -d ' ') lines)"

# 3. Optical Recognition of Handwritten Digits - Test Set (1,797 samples, 8x8 pixels, 10 classes)
OPTDIGITS_TES_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
echo "[3/3] Downloading 8x8 Optical Digits (Test Set)..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "${OPTDIGITS_TES_URL}" -o "${DATA_DIR}/optdigits.tes"
elif command -v wget >/dev/null 2>&1; then
    wget -q "${OPTDIGITS_TES_URL}" -O "${DATA_DIR}/optdigits.tes"
fi
echo "  -> Saved to ${DATA_DIR}/optdigits.tes ($(wc -l < "${DATA_DIR}/optdigits.tes" | tr -d ' ') lines)"

echo ""
echo "All datasets downloaded successfully into ${DATA_DIR}/!"
