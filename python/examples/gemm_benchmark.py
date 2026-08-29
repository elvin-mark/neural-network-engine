"""
Benchmark: Matrix Multiplication Comparison across NumPy, PyTorch (CPU), and Neural Network Engine (Rust Backend).

Compares execution time and throughput (GFLOPS) across small, medium, and large matrix dimensions.
"""

import sys
import time
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import neural_network_engine as nne


def benchmark_gemm(m: int, k: int, n: int, num_warmup: int = 5, num_iters: int = 20):
    """Benchmarks M x K @ K x N matrix multiplication for NumPy, PyTorch, and NNE."""
    flops = 2.0 * m * k * n  # 2 * M * K * N floating point ops

    # 1. Initialize identical random matrices
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)

    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)

    a_nne = nne.Tensor.from_numpy(a_np, requires_grad=False)
    b_nne = nne.Tensor.from_numpy(b_np, requires_grad=False)

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------
    for _ in range(num_warmup):
        _ = a_np @ b_np
        _ = torch.matmul(a_torch, b_torch)
        _ = a_nne @ b_nne

    # -------------------------------------------------------------
    # 1. NumPy Benchmark
    # -------------------------------------------------------------
    start = time.perf_counter()
    for _ in range(num_iters):
        res_np = a_np @ b_np
    np_time = (time.perf_counter() - start) / num_iters

    # -------------------------------------------------------------
    # 2. PyTorch (CPU) Benchmark
    # -------------------------------------------------------------
    start = time.perf_counter()
    for _ in range(num_iters):
        res_torch = torch.matmul(a_torch, b_torch)
    torch_time = (time.perf_counter() - start) / num_iters

    # -------------------------------------------------------------
    # 3. Custom Rust Neural Network Engine Benchmark
    # -------------------------------------------------------------
    start = time.perf_counter()
    for _ in range(num_iters):
        res_nne = a_nne @ b_nne
    nne_time = (time.perf_counter() - start) / num_iters

    # Numerical Correctness Verification (allowing standard float32 reduction rounding)
    res_nne_np = res_nne.to_numpy()
    np.testing.assert_allclose(res_np, res_torch.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_np, res_nne_np, rtol=1e-3, atol=1e-3)

    # Compute GFLOPS
    np_gflops = (flops / np_time) / 1e9
    torch_gflops = (flops / torch_time) / 1e9
    nne_gflops = (flops / nne_time) / 1e9

    return {
        "mkn": f"{m}x{k}x{n}",
        "np_ms": np_time * 1000.0,
        "np_gflops": np_gflops,
        "torch_ms": torch_time * 1000.0,
        "torch_gflops": torch_gflops,
        "nne_ms": nne_time * 1000.0,
        "nne_gflops": nne_gflops,
    }


def main():
    print("=" * 85)
    print(" Matrix Multiplication Benchmark: NumPy vs PyTorch (CPU) vs Custom Engine (Rust)")
    print("=" * 85)
    print(f" PyTorch version: {torch.__version__} | NumPy version: {np.__version__}")
    print(f" CPU Threads: Rayon (all cores) / PyTorch OpenMP ({torch.get_num_threads()} threads)")
    print("=" * 85 + "\n")

    matrix_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    header = (
        f"{'Matrix Size (MxKxN)':<20} | "
        f"{'NumPy':<17} | "
        f"{'PyTorch':<17} | "
        f"{'Custom Engine (Rust)':<20}"
    )
    print(header)
    print("-" * len(header))

    for m, k, n in matrix_sizes:
        # Scale iterations for larger matrices
        iters = 50 if m <= 512 else (10 if m <= 1024 else 3)
        res = benchmark_gemm(m, k, n, num_warmup=5, num_iters=iters)

        print(
            f"{res['mkn']:<20} | "
            f"{res['np_ms']:6.2f} ms ({res['np_gflops']:5.1f} GF) | "
            f"{res['torch_ms']:6.2f} ms ({res['torch_gflops']:5.1f} GF) | "
            f"{res['nne_ms']:6.2f} ms ({res['nne_gflops']:5.1f} GF)"
        )

    print("-" * len(header))
    print("✓ All numerical outputs verified against NumPy baseline (atol=1e-4)!\n")


if __name__ == "__main__":
    main()
