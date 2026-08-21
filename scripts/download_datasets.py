#!/usr/bin/env python3
"""
Downloads Fisher's Iris and UCI Optical Handwritten Digits datasets into the data/ directory.
Zero third-party dependencies (uses Python standard library urllib).
"""

import os
import sys
import urllib.request
from pathlib import Path

DATASETS = [
    (
        "iris.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "Fisher's Iris Dataset (150 samples)",
    ),
    (
        "optdigits.tra",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "UCI 8x8 Optical Digits - Training (3,823 samples)",
    ),
    (
        "optdigits.tes",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        "UCI 8x8 Optical Digits - Test (1,797 samples)",
    ),
]

def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   Downloading Toy Datasets for Neural Network Engine")
    print("=" * 60)
    print(f"Target directory: {data_dir}\n")

    for idx, (filename, url, description) in enumerate(DATASETS, 1):
        target_path = data_dir / filename
        print(f"[{idx}/{len(DATASETS)}] Downloading {description}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NeuralNetworkEngine-Downloader/1.0"},
            )
            with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
                out_file.write(response.read())

            line_count = sum(1 for _ in open(target_path, "r", encoding="utf-8", errors="ignore"))
            print(f"  -> Saved {target_path.name} ({line_count} lines)")
        except Exception as e:
            print(f"  -> Error downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nAll datasets downloaded successfully into {data_dir}!")

if __name__ == "__main__":
    main()
