# ResNet Architecture (`models::resnet`)

Residual convolutional neural networks implementing skip connections to eliminate vanishing gradients in deep computer vision pipelines.

**Source files**:
- Core implementation: [`src/models/resnet.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/models/resnet.rs)
- Unit tests: [`tests/resnet_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/resnet_tests.rs)
- Example: [`examples/17_resnet_cifar_vision.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/examples/17_resnet_cifar_vision.rs)

---

## Residual Blocks

```mermaid
flowchart LR
    subgraph BasicBlock (ResNet-18, ResNet-34)
        X1["Input x"] --> C1["Conv 3×3 + BN + ReLU"]
        C1 --> C2["Conv 3×3 + BN"]
        X1 --> Shortcut1["Shortcut (Identity or 1×1 Conv)"]
        C2 & Shortcut1 --> Add1["Add + ReLU"]
    end

    subgraph BottleneckBlock (ResNet-50, ResNet-101)
        X2["Input x"] --> B1["Conv 1×1 (Reduce) + BN + ReLU"]
        B1 --> B2["Conv 3×3 + BN + ReLU"]
        B2 --> B3["Conv 1×1 (Expand 4×) + BN"]
        X2 --> Shortcut2["Shortcut (Identity or 1×1 Conv)"]
        B3 & Shortcut2 --> Add2["Add + ReLU"]
    end
```

### 1. `BasicBlock`
- Used in shallow ResNet models (18 and 34 layers).
- Contains two $3 \times 3$ convolutions with `BatchNorm2d` and residual addition:
  $$y = \text{ReLU}(\mathcal{F}(x, \{W_i\}) + W_s x)$$

### 2. `BottleneckBlock`
- Used in deep ResNet models (50, 101, 152 layers).
- Three-stage bottleneck design:
  1. $1 \times 1$ convolution to reduce channel dimension.
  2. $3 \times 3$ spatial convolution.
  3. $1 \times 1$ convolution to restore/expand channel dimension by factor of 4.

---

## Standard Presets

```rust
// ResNet-18: [2, 2, 2, 2] BasicBlocks
pub fn resnet18(num_classes: usize) -> ResNet

// ResNet-34: [3, 4, 6, 3] BasicBlocks
pub fn resnet34(num_classes: usize) -> ResNet

// ResNet-50: [3, 4, 6, 3] BottleneckBlocks
pub fn resnet50(num_classes: usize) -> ResNet
```

---

## See Also
- [Convolution & GEMM Runtime](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/matmul-and-convolution.md)
- [Normalization (`BatchNorm2d`)](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/normalization.md)
- [Vision Transformers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/vit.md)
