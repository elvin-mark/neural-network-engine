# Python Bindings & NumPy Interop (`python`)

High-performance Python extension module powered by PyO3 providing seamless bidirectional zero-copy array interoperability with NumPy.

**Source files**:
- Core implementation: [`src/python.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/python.rs)
- Python Package Configuration: [`pyproject.toml`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/pyproject.toml)
- Examples: [`python/examples/`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/python/examples/)

---

## Building the Extension with Maturin

The Python extension is built via [Maturin](https://github.com/PyO3/maturin):

```bash
# In the repository root:
maturin develop --release --features python
```

---

## Python API Overview

The module exports `neural_network_engine` to Python:

```python
import numpy as np
import neural_network_engine as nne

# 1. Create a tensor from NumPy ndarray
np_arr = np.random.randn(4, 128).astype(np.float32)
tensor = nne.Tensor.from_numpy(np_arr)

# 2. Differentiable operations
weights = nne.Tensor.randn([128, 64], requires_grad=True)
out = tensor.matmul(weights).relu()

# 3. Backward pass
loss = out.sum()
loss.backward()

# 4. Extract gradients back to NumPy
grad_np = weights.grad.to_numpy()
print("Weight gradient shape:", grad_np.shape)
```

---

## Exposed Classes & Functions

- **`Tensor`**: Wraps Rust `autograd::Tensor`. Supports arithmetic operators (`+`, `-`, `*`, `/`), matrix multiplication (`matmul`), activations (`relu`, `gelu`, `silu`), reductions (`sum`, `mean`), and `.backward()`.
- **`from_numpy(arr)` / `.to_numpy()`**: Bidirectional conversion between NumPy `np.ndarray` and Rust `Tensor`.
- **`Linear`**, **`Conv2d`**, **`Sequential`**: Standard layers callable directly from Python scripts.
- **`SGD`**, **`Adam`**: Optimizers exposing `.step()` and `.zero_grad()`.

---

## See Also
- [Core Tensor Runtime](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/tensor.md)
- [Automatic Differentiation Engine](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/autograd.md)
