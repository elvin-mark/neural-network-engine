# Dynamic Autograd Engine

The automatic differentiation subsystem implements dynamic reverse-mode automatic differentiation (Autograd) over tape-free computation graphs (DAGs) with automatic gradient accumulation and broadcast reduction.

**Source files**:
- Public Interface: [`src/autograd/tensor.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/autograd/tensor.rs)
- Graph Node & Dispatch: [`src/autograd/node.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/autograd/node.rs)
- Context & Gradient Guards: [`src/autograd/context.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/autograd/context.rs)
- Numerical Verification: [`src/utils/gradcheck.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/utils/gradcheck.rs)

---

## The `Tensor` Wrapper & Node Graph

A user-facing `Tensor` wraps an `Arc<RwLock<TensorInner>>`:

```rust
pub struct Tensor {
    inner: Arc<RwLock<TensorInner>>,
}

pub struct TensorInner {
    pub data: RawTensor,
    pub grad: Option<RawTensor>,
    pub requires_grad: bool,
    pub parents: Vec<Tensor>,
    pub backward_fn: Option<BackwardFn>,
    pub id: usize,
}
```

- **`requires_grad`**: Determines whether operations consuming this tensor should track history and register a backward closure.
- **`backward_fn`**: A closure taking the output gradient $\frac{\partial L}{\partial y}$ and returning a list of input gradients $\left[\frac{\partial L}{\partial x_0}, \frac{\partial L}{\partial x_1}, \dots\right]$.

---

## Forward Execution & Backward Traversal

```mermaid
flowchart TD
    subgraph Forward Pass
        X1["Tensor X1 (requires_grad: true)"] --> Op["Forward Operation: Y = X1 * W + B"]
        W["Tensor W (requires_grad: true)"] --> Op
        B["Tensor B (requires_grad: true)"] --> Op
        Op --> Y["Tensor Y"]
    end

    subgraph Backward Pass (Topological Order)
        Loss["Loss.backward()"] --> DY["dY = 1.0 (seed gradient)"]
        DY --> BwdY["Y.backward_fn(dY)"]
        BwdY --> DX1["Accumulate dX1 into X1.grad"]
        BwdY --> DW["Accumulate dW into W.grad"]
        BwdY --> DB["Accumulate dB into B.grad"]
    end
```

### 1. Topological Sorting
When `loss.backward()` is called:
1. The engine collects all reachable nodes via Depth-First Search (DFS) post-order traversal and reverses it to produce a valid topological ordering.
2. The initial gradient is seeded with `1.0` (matching scalar loss).

### 2. In-Place Gradient Accumulation
When a node produces multiple branches in the graph, gradients from downstream dependents are accumulated in-place into `grad`:
$$x.\text{grad} \leftarrow x.\text{grad} + \frac{\partial L}{\partial x}$$

### 3. Broadcast Gradient Reduction (`unbroadcast_to`)
If a tensor was expanded during forward broadcasting (e.g. adding a bias $[D]$ to activations $[B, D]$), the incoming gradient $[B, D]$ must be collapsed back to $[D]$:

```rust
pub fn unbroadcast_to(grad: &RawTensor, target_shape: &[usize]) -> Result<RawTensor>
```
The algorithm:
1. Sums out all extra leading dimensions ($N_{\text{grad}} > N_{\text{target}}$).
2. Sums out any axes where $\text{target}_{\text{shape}}[i] = 1$ while $\text{grad}_{\text{shape}}[i] > 1$ with `keepdim = true`.

---

## Differentiable Operations & Derivatives

| Operation | Forward Formulation | Input Gradients |
|---|---|---|
| **Addition** ($Z = X + Y$) | $Z = X + Y$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}$, $\frac{\partial L}{\partial Y} = \frac{\partial L}{\partial Z}$ |
| **Subtraction** ($Z = X - Y$) | $Z = X - Y$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}$, $\frac{\partial L}{\partial Y} = -\frac{\partial L}{\partial Z}$ |
| **Hadamard Product** ($Z = X \odot Y$) | $Z = X \odot Y$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \odot Y$, $\frac{\partial L}{\partial Y} = \frac{\partial L}{\partial Z} \odot X$ |
| **Matrix Multiplication** ($C = A B$) | $C = A B$ | $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$, $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$ |
| **Convolution 2D** | $Y = \text{Conv2d}(X, W)$ | $\frac{\partial L}{\partial X} = \text{col2im}(W^T \frac{\partial L}{\partial Y})$, $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \text{im2col}(X)^T$ |
| **ReLU** | $Y = \max(0, X)$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot \mathbb{I}(X > 0)$ |
| **GELU** | $Y = X \cdot \Phi(X)$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \left(\Phi(X) + X \cdot \phi(X)\right)$ |
| **Sum Reduction** | $y = \sum_i X_i$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial y} \cdot \mathbf{1}_{\text{shape}(X)}$ |

---

## Gradient Control Contexts

During inference or evaluation, graph construction overhead is bypassed using thread-local gradient guards:

```rust
// Scoped closure:
let output = no_grad(|| model.forward(&input));

// Explicit guard:
{
    let _guard = NoGradGuard::new();
    let pred = model.forward(&input);
}
```

---

## Numerical Verification (`gradcheck`)

To guarantee gradient correctness across every custom layer, the engine includes a finite-difference verification utility:

```rust
pub fn gradcheck<F>(f: F, inputs: &[Tensor], eps: f32, tol: f32) -> Result<bool>
```

It approximates the Jacobian via two-sided symmetric central differences:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}$$
and asserts that the maximum relative error compared to analytical backpropagation gradients satisfies:
$$\frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|) + 10^{-7}} < \text{tol}$$

---

## See Also
- [Tensor Runtime (`RawTensor`)](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/tensor.md)
- [Optimizers & Training Loop](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/optimizers.md)
- [Neural Network Modules](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/modules.md)
