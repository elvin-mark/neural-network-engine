# Normalization Layers

State-of-the-art normalization modules for vision models, recurrent architectures, and modern large language models.

**Source files**:
- Core implementation: [`src/nn/norm.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/norm.rs)
- GPU Kernels: [`src/gpu/layers.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/gpu/layers.rs)

---

## Comparison Table

| Layer | Normalization Dimension | Primary Domain | Running Stats | Affine Parameters |
|---|---|---|---|---|
| **`LayerNorm`** | Feature axis ($D$) | Transformers (BERT, GPT-2, ViT) | No (computed per-sample) | $\gamma, \beta$ |
| **`RMSNorm`** | Feature axis ($D$) | Modern LLMs (LLaMA, Mistral) | No (scale-only) | $\gamma$ (no mean, no bias) |
| **`BatchNorm1d`** | Batch axis ($N$) | Dense / MLP Networks | Yes (running mean/variance) | $\gamma, \beta$ |
| **`BatchNorm2d`** | Batch & Spatial ($N, H, W$) | Convnets (ResNet) | Yes (running mean/variance) | $\gamma, \beta$ |

---

## Mathematical Formulations

### 1. `LayerNorm`
Normalizes across the last dimension $D$:
$$\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

```rust
let ln = LayerNorm::new(vec![768], 1e-5, true); // normalized_shape, eps, affine
```

### 2. `RMSNorm` (Root Mean Square Normalization)
Removes the mean calculation $\mu$ and centering step, computing only the root mean square:
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$$
$$y_i = \gamma_i \frac{x_i}{\text{RMS}(x)}$$

RMSNorm provides a 10%–50% speedup over standard LayerNorm while preserving identical training stability in Transformer architectures.

```rust
let rms = RMSNorm::new(4096, 1e-6); // dim, eps
```

### 3. `BatchNorm1d` & `BatchNorm2d`
Normalizes across batch and spatial channels:
- **Training Mode**: Computes sample mean $\mu_{\mathcal{B}}$ and variance $\sigma^2_{\mathcal{B}}$, and updates running statistics via exponential moving average:
  $$\mu_{\text{run}} \leftarrow (1 - m) \mu_{\text{run}} + m \cdot \mu_{\mathcal{B}}$$
  $$\sigma^2_{\text{run}} \leftarrow (1 - m) \sigma^2_{\text{run}} + m \cdot \sigma^2_{\mathcal{B}}$$
  (Default momentum $m = 0.1$)
- **Evaluation Mode**: Evaluates using pre-computed $\mu_{\text{run}}$ and $\sigma^2_{\text{run}}$, removing inter-sample dependency during inference.

```rust
let bn2d = BatchNorm2d::new(64, 1e-5, 0.1, true); // channels, eps, momentum, affine
```

---

## See Also
- [Neural Network Modules](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/modules.md)
- [Transformer Layers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/attention-and-transformers.md)
- [ResNet Architecture](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/resnet.md)
