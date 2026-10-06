# Activation Functions

Non-linear activation functions with numerically stable forward passes and exact analytical derivatives.

**Source files**:
- Core implementation: [`src/nn/activations.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/activations.rs)
- Autograd Bindings: [`src/autograd/tensor.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/autograd/tensor.rs)

---

## Supported Activations Summary

| Name | Mathematical Definition | Derivative $\frac{dy}{dx}$ | Numerical Stability Notes |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | $\mathbb{I}(x > 0)$ | Fast branchless execution |
| **LeakyReLU** | $\max(\alpha x, x)$ | $\begin{cases} 1 & x > 0 \\ \alpha & x \le 0 \end{cases}$ | Configurable slope $\alpha$ (default: 0.01) |
| **GELU** | $x \cdot \Phi(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$ | $\Phi(x) + x \phi(x)$ | High-precision polynomial approximation |
| **SiLU / Swish** | $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ | $\sigma(x) + x \sigma(x) (1 - \sigma(x))$ | Clamped negative exponentials to prevent overflow |
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $\sigma(x) (1 - \sigma(x))$ | Safe range clamping for $x < -80$ and $x > 80$ |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | Direct platform SIMD support |
| **Softmax** | $\frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ | $S_i (\delta_{ij} - S_j)$ | Subtraction of maximum ($\max x$) prevents $e^x$ overflow |
| **SwiGLU** | $\text{SwiGLU}(x, W, V) = (x W \cdot \sigma(x W)) \odot (x V)$ | Exact multi-branch backward | Decoupled gate and linear projections |

---

## Detailed Formulations

### GELU (Gaussian Error Linear Unit)
Used across modern transformers (BERT, GPT-2, ViT):
$$\text{GELU}(x) = 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### Softmax with Max-Subtraction
For arbitrary multidimensional inputs, softmax along axis $d$ operates via:
$$m = \max_{j} x_j$$
$$p_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$
This mathematical identity prevents `NaN` values when logits exceed floating-point exponents ($x > 88.7$ in IEEE 754 single precision).

---

## Struct vs Functional API

Each activation can be used both as a stateful `Module` struct in sequential models or functionally directly on a `Tensor`:

```rust
// 1. As a Module:
let act = GELU;
let y = act.forward(&x)?;

// 2. Functionally on Tensor:
let y = x.relu()?;
let y = x.gelu()?;
let y = x.silu()?;
let y = x.softmax(-1)?;
```

---

## See Also
- [Neural Network Modules](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/modules.md)
- [Loss Functions](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/losses.md)
- [Transformer Architectures](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/attention-and-transformers.md)
