"""
Python Example 02: ResNet-18 Computer Vision Training on CIFAR Images.

Demonstrates:
1. Building a deep Convolutional Residual Network (ResNet-18) in Python.
2. 2D Convolutions, 2D Batch Normalization, and Residual Skip Connections.
3. Multiclass classification training with CrossEntropyLoss and Adam.
"""

import sys
import time
from pathlib import Path

# Add python module directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_network_engine as nne
import numpy as np


def main():
    print("=" * 65)
    print(" Python Example 02: ResNet-18 Vision Training with Rust Backend")
    print("=" * 65 + "\n")

    # 1. Generate Synthetic 32x32 RGB Image Batch (CIFAR-10 style)
    np.random.seed(42)
    batch_size = 8
    in_channels = 3
    height, width = 32, 32
    num_classes = 10

    print("Dataset Configuration:")
    print(f"  • Batch: {batch_size} images")
    print(f"  • Image Dimensions: {in_channels}x{height}x{width} (RGB)")
    print(f"  • Classes: {num_classes} (CIFAR-10)\n")

    X_np = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
    Y_np = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.float32)

    X = nne.Tensor.from_numpy(X_np, requires_grad=False)
    Y = nne.Tensor.from_numpy(Y_np, requires_grad=False)

    # 2. Instantiate ResNet-18
    resnet = nne.ResNet18(num_classes=num_classes, in_channels=in_channels, cifar_stem=True)
    params = resnet.parameters()
    print(f"ResNet-18 initialized with {len(params)} trainable parameter tensors.")

    # 3. Setup Optimizer & Loss
    optimizer = nne.Adam(params, lr=1e-3)
    loss_fn = nne.CrossEntropyLoss()

    print("Training ResNet-18 for 15 optimization steps...\n")

    start_time = time.time()
    for step in range(1, 16):
        optimizer.zero_grad()

        # Forward pass through Conv -> BatchNorm -> Residual Layers -> Global Pool -> Linear Head
        logits = resnet(X)
        loss = loss_fn(logits, Y)

        # Autograd backward through the deep residual graph
        loss.backward()

        # Optimizer step
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:2d}/15 | CrossEntropyLoss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\nResNet-18 training completed in {elapsed:.2f}s ({elapsed * 1000 / 15:.1f} ms/step)!")
    print("Python ResNet-18 vision example executed successfully!\n")


if __name__ == "__main__":
    main()
