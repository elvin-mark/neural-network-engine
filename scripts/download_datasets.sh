#!/usr/bin/env bash
# Download canonical datasets: Iris, 8x8 Digits, MNIST, CIFAR-10, and CIFAR-100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "============================================================"
echo "   Downloading Datasets for Neural Network Engine"
echo "============================================================"
echo "Target directory: ${DATA_DIR}"
echo ""

fetch() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -sSL "${url}" -o "${output}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "${url}" -O "${output}"
    else
        echo "Error: Neither curl nor wget found on system." >&2
        exit 1
    fi
}

# 1. Fisher's Iris Dataset (150 samples)
echo "[1/5] Downloading Fisher's Iris Dataset..."
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" "${DATA_DIR}/iris.data"
echo "  -> Saved iris.data"

# 2. 8x8 Optical Digits (3,823 train + 1,797 test)
echo "[2/5] Downloading 8x8 Optical Digits..."
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra" "${DATA_DIR}/optdigits.tra"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes" "${DATA_DIR}/optdigits.tes"
echo "  -> Saved optdigits.tra and optdigits.tes"

# 3. MNIST 28x28 Handwritten Digits (60,000 train + 10,000 test)
echo "[3/5] Downloading 28x28 MNIST Dataset..."
MNIST_BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
for f in "train-images-idx3-ubyte" "train-labels-idx1-ubyte" "t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte"; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        fetch "${MNIST_BASE}/${f}.gz" "${DATA_DIR}/${f}.gz"
        gzip -df "${DATA_DIR}/${f}.gz"
    fi
done
echo "  -> Saved MNIST IDX binary files"

# 4. CIFAR-10 Binary Dataset (60,000 3x32x32 color images across 10 classes)
echo "[4/5] Downloading CIFAR-10 Dataset..."
if [ ! -d "${DATA_DIR}/cifar-10-batches-bin" ]; then
    fetch "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" "${DATA_DIR}/cifar-10-binary.tar.gz"
    tar -xzf "${DATA_DIR}/cifar-10-binary.tar.gz" -C "${DATA_DIR}"
    rm -f "${DATA_DIR}/cifar-10-binary.tar.gz"
fi
echo "  -> Saved CIFAR-10 binary batches in data/cifar-10-batches-bin/"

# 5. CIFAR-100 Binary Dataset (60,000 3x32x32 color images across 100 classes)
echo "[5/5] Downloading CIFAR-100 Dataset..."
if [ ! -d "${DATA_DIR}/cifar-100-binary" ]; then
    fetch "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz" "${DATA_DIR}/cifar-100-binary.tar.gz"
    tar -xzf "${DATA_DIR}/cifar-100-binary.tar.gz" -C "${DATA_DIR}"
    rm -f "${DATA_DIR}/cifar-100-binary.tar.gz"
fi
echo "  -> Saved CIFAR-100 binary batches in data/cifar-100-binary/"

echo ""
echo "All datasets downloaded and unpacked successfully in ${DATA_DIR}/!"
