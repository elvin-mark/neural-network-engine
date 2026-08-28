"""
Example: End-to-End Neural Network Training in Python using Rust Backend.

Demonstrates:
1. Converting NumPy arrays to Rust Tensors with autograd enabled.
2. Building and training a deep MLP on synthetic non-linear classification data.
3. PyO3 zero-copy tensors, forward execution, autograd backward, and Adam optimization.
"""

import sys
import time
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_network_engine as nne
import numpy as np


def main():
    print("=" * 60)
    print(" Python Bindings: Deep Learning in Python with Pure Rust Engine")
    print("=" * 60 + "\n")

    # 1. Generate Synthetic 2D Classification Data (Two Moons / Spiral)
    np.random.seed(42)
    n_samples = 200
    in_features = 16
    hidden_features = 64
    out_features = 4

    print(f"Generating synthetic dataset: {n_samples} samples, {in_features} features -> {out_features} classes")
    X_np = np.random.randn(n_samples, in_features).astype(np.float32)
    Y_np = np.random.randn(n_samples, out_features).astype(np.float32)

    # 2. Convert NumPy Arrays to Rust Tensors
    X = nne.Tensor.from_numpy(X_np, requires_grad=False)
    Y = nne.Tensor.from_numpy(Y_np, requires_grad=False)

    print(f"  • Input Tensor:  shape={X.shape}, dtype=float32")
    print(f"  • Target Tensor: shape={Y.shape}, dtype=float32\n")

    # 3. Define Deep Architecture in Python using Rust Neural Network Modules
    fc1 = nne.Linear(in_features, hidden_features)
    ln1 = nne.LayerNorm(hidden_features)
    gelu = nne.GELU()
    fc2 = nne.Linear(hidden_features, out_features)

    # Collect all trainable parameters
    params = []
    params.extend(fc1.parameters())
    params.extend(ln1.parameters())
    params.extend(fc2.parameters())

    print(f"Model initialized with {len(params)} parameter tensors.")

    # 4. Initialize Adam Optimizer
    optimizer = nne.Adam(params, lr=0.01)
    scaler = nne.LossScaler(1024.0)

    print(f"Training with Adam optimizer (lr=0.01) and AMP LossScaler (scale={scaler.current_scale()})...\n")

    # 5. Training Loop
    start_time = time.time()
    epochs = 50

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward Pass: Rust executes matmul, layernorm, GELU at full speed
        h1 = gelu(ln1(fc1(X)))
        pred = fc2(h1)

        # MSE Loss: (pred - Y)^2.mean()
        diff = pred - Y
        loss = (diff * diff).mean()

        # Backward Pass via Rust Autograd DAG
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Optimizer Step
        scaler.step_adam(optimizer)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.6f} | Scale: {scaler.current_scale():.1f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed * 1000:.2f} ms ({elapsed * 1000 / epochs:.3f} ms/epoch)!")

    # 6. Convert Final Predictions to NumPy Array
    final_pred_np = pred.to_numpy()
    print(f"Final predictions exported to NumPy array: shape={final_pred_np.shape}, type={type(final_pred_np)}")
    print("\nPython bindings demo executed successfully!")


if __name__ == "__main__":
    main()
