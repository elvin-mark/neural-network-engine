# Neural Network Modules (`nn::Module`)

Composable neural network layers and containers implementing stateful forward execution and parameter registration.

**Source files**:
- Module Trait: [`src/nn/module.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/module.rs)
- Linear: [`src/nn/linear.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/linear.rs)
- Convolution: [`src/nn/conv.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/conv.rs)
- Pooling: [`src/nn/pooling.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/pooling.rs)
- Dropout: [`src/nn/dropout.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/dropout.rs)
- Embedding: [`src/nn/embedding.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/embedding.rs)
- Sequential: [`src/nn/sequential.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/sequential.rs)

---

## The `Module` Trait

All layers implement the unified `Module` trait:

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
    fn set_training(&mut self, _mode: bool) {}
    fn is_training(&self) -> bool { true }
}
```

- **`parameters()`**: Returns a flat list of learnable weight and bias tensors to pass directly to an `Optimizer`.
- **`set_training(mode)`**: Toggles training mode (affecting layers with training-specific behavior like `Dropout` and `BatchNorm`).

---

## Core Layers Overview

### 1. `Linear` (Dense / Fully-Connected)
Computes affine transformation:
$$Y = X W^T + B$$
- **Weights Shape**: `[out_features, in_features]`
- **Bias Shape**: `[out_features]` (optional)
- **Input**: `[*, in_features]`
- **Output**: `[*, out_features]`

```rust
let fc = Linear::new(784, 128, true); // in_features, out_features, bias
```

### 2. `Conv2d` (2D Spatial Convolution)
Computes spatial cross-correlation over 4D feature maps $[N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}}]$ using lowered parallel GEMM (`im2col`):
- **Kernel Shape**: `[out_channels, in_channels / groups, kernel_h, kernel_w]`
- **Bias Shape**: `[out_channels]`
- Supports custom `stride`, `padding`, `dilation`, and `groups`.

```rust
let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true);
```

### 3. `MaxPool2d`
Downsamples spatial dimensions by computing window maximums:
- **Input**: `[N, C, H, W]`
- **Output**: `[N, C, H_out, W_out]`

```rust
let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0));
```

### 4. `Dropout`
Applies inverted dropout regularization:
- **Training**: Zeroes elements with probability $p$ and scales surviving activations by $\frac{1}{1 - p}$.
- **Evaluation**: Identity transformation ($Y = X$).

```rust
let dropout = Dropout::new(0.5);
```

### 5. `Embedding`
Dense lookup table converting discrete token indices into embedding vectors:
- **Weights Shape**: `[num_embeddings, embedding_dim]`
- **Input**: `[batch_size, seq_len]` (indices)
- **Output**: `[batch_size, seq_len, embedding_dim]`

```rust
let emb = Embedding::new(32000, 4096);
```

### 6. `Sequential`
Chains multiple modules in linear execution order:

```rust
let model = Sequential::new(vec![
    Box::new(Linear::new(784, 256, true)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)),
    Box::new(Linear::new(256, 10, true)),
]);
```

---

## See Also
- [Activation Functions](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/activations.md)
- [Normalization Layers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/normalization.md)
- [Loss Functions](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/losses.md)
