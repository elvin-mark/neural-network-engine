#!/usr/bin/env python3
"""
Downloads Fisher's Iris, 8x8 Digits, MNIST, CIFAR-10, and CIFAR-100 datasets into the data/ directory.
Zero third-party dependencies (uses Python standard libraries urllib, gzip, tarfile).
"""

import gzip
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DATASETS = [
    (
        "iris.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "Fisher's Iris Dataset (150 samples)",
        "file",
    ),
    (
        "optdigits.tra",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "UCI 8x8 Optical Digits - Training (3,823 samples)",
        "file",
    ),
    (
        "optdigits.tes",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        "UCI 8x8 Optical Digits - Test (1,797 samples)",
        "file",
    ),
    (
        "train-images-idx3-ubyte",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "MNIST 28x28 Digits - Training Images (60,000 samples)",
        "gzip",
    ),
    (
        "train-labels-idx1-ubyte",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "MNIST 28x28 Digits - Training Labels (60,000 samples)",
        "gzip",
    ),
    (
        "t10k-images-idx3-ubyte",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "MNIST 28x28 Digits - Test Images (10,000 samples)",
        "gzip",
    ),
    (
        "t10k-labels-idx1-ubyte",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "MNIST 28x28 Digits - Test Labels (10,000 samples)",
        "gzip",
    ),
    (
        "cifar-10-batches-bin",
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "CIFAR-10 Binary Batches (60,000 3x32x32 RGB images across 10 classes)",
        "tar",
    ),
    (
        "cifar-100-binary",
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        "CIFAR-100 Binary Batches (60,000 3x32x32 RGB images across 100 classes)",
        "tar",
    ),
    (
        "tinystories.txt",
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "TinyStories Corpus (19MB plain text stories)",
        "raw",
    ),
]


def download_file(url: str, target: Path):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NeuralNetworkEngine-Downloader/1.0"},
    )
    with urllib.request.urlopen(req) as response, open(target, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   Downloading Datasets for Neural Network Engine")
    print("=" * 60)
    print(f"Target directory: {data_dir}\n")

    for idx, (filename, url, description, kind) in enumerate(DATASETS, 1):
        target_path = data_dir / filename
        if target_path.exists():
            print(f"[{idx}/{len(DATASETS)}] {description} already exists, skipping.")
            continue

        print(f"[{idx}/{len(DATASETS)}] Downloading {description}...")
        try:
            if kind == "file":
                download_file(url, target_path)
                print(f"  -> Saved {target_path.name}")
            elif kind == "gzip":
                temp_gz = data_dir / (filename + ".gz")
                download_file(url, temp_gz)
                with gzip.open(temp_gz, "rb") as gz_in, open(target_path, "wb") as out_file:
                    shutil.copyfileobj(gz_in, out_file)
                temp_gz.unlink(missing_ok=True)
                print(f"  -> Unpacked {target_path.name}")
            elif kind == "tar":
                temp_tar = data_dir / (filename + ".tar.gz")
                download_file(url, temp_tar)
                with tarfile.open(temp_tar, "r:gz") as tar_in:
                    tar_in.extractall(path=data_dir)
                temp_tar.unlink(missing_ok=True)
                print(f"  -> Extracted {filename}/")
        except Exception as e:
            print(f"  -> Error downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nAll datasets downloaded successfully into {data_dir}!")


if __name__ == "__main__":
    main()
