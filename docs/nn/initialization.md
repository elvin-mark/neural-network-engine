# Weight Initialization (`nn::init`)

Mathematically principled weight initialization algorithms to maintain stable variance of activations and gradients across deep networks.

**Source files**:
- Core implementation: [`src/nn/init.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/init.rs)
- Unit tests: [`tests/init_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/init_tests.rs)

---

## Fan In and Fan Out Computation

For a tensor with shape $[D_{\text{out}}, D_{\text{in}}, K_1, K_2, \dots]$, the receptive field spatial size is:
$$R = \prod_{i=1}^M K_i$$
- **$\text{fan\_in} = D_{\text{in}} \times R$**: Number of input connections per neuron.
- **$\text{fan\_out} = D_{\text{out}} \times R$**: Number of output connections per neuron.

```rust
pub fn calculate_fan_in_and_fan_out(tensor: &Tensor) -> Result<(usize, usize)>
```

---

## Gain Values for Non-Linearities

```rust
pub enum NonLinearity {
    Linear,
    Conv,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f32),
    GELU,
    SiLU,
}
```

The gain factor scales the standard deviation according to the non-linearity:
$$\text{gain}(\text{Linear}) = 1.0, \quad \text{gain}(\text{Tanh}) = \frac{5}{3} \approx 1.667, \quad \text{gain}(\text{ReLU}) = \sqrt{2} \approx 1.414$$
$$\text{gain}(\text{LeakyReLU}(\alpha)) = \sqrt{\frac{2}{1 + \alpha^2}}$$

---

## Supported Initializers

### 1. Xavier / Glorot Initialization
Maintains activation and backpropagation variance under linear/symmetric activations (Tanh, Sigmoid):

- **Uniform**: $\mathcal{U}(-a, a)$ with $a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$
  ```rust
  pub fn xavier_uniform_(tensor: &mut Tensor, gain: f32) -> Result<()>
  ```
- **Normal**: $\mathcal{N}(0, \sigma^2)$ with $\sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$
  ```rust
  pub fn xavier_normal_(tensor: &mut Tensor, gain: f32) -> Result<()>
  ```

### 2. Kaiming / He Initialization
Optimized for non-symmetric rectifying activations (ReLU, LeakyReLU, GELU):

- **Uniform**: $\mathcal{U}(-a, a)$ with $a = \text{gain} \times \sqrt{\frac{3}{\text{fan}}}$
- **Normal**: $\mathcal{N}(0, \sigma^2)$ with $\sigma = \frac{\text{gain}}{\sqrt{\text{fan}}}$
- **Modes**:
  - `FanMode::FanIn`: Preserves magnitude of activation variance in forward pass.
  - `FanMode::FanOut`: Preserves gradient variance during backward pass.

```rust
pub fn kaiming_uniform_(tensor: &mut Tensor, a: f32, mode: FanMode, nonlinearity: NonLinearity) -> Result<()>
pub fn kaiming_normal_(tensor: &mut Tensor, a: f32, mode: FanMode, nonlinearity: NonLinearity) -> Result<()>
```

### 3. Orthogonal Initialization
Generates a semi-orthogonal or orthogonal matrix via QR decomposition of a random Gaussian matrix, preventing vanishing or exploding gradients in recurrent networks (RNN, LSTM, GRU):

```rust
pub fn orthogonal_(tensor: &mut Tensor, gain: f32) -> Result<()>
```

### 4. Basic Initializers
- `zeros_(tensor)`: Fills tensor with $0.0$.
- `ones_(tensor)`: Fills tensor with $1.0$.
- `constant_(tensor, val)`: Fills tensor with constant value.
- `uniform_(tensor, from, to)`: Samples from $\mathcal{U}(\text{from}, \text{to})$.
- `normal_(tensor, mean, std)`: Samples from $\mathcal{N}(\mu, \sigma^2)$.

---

## See Also
- [Neural Network Modules](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/modules.md)
- [Recurrent Layers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/recurrent.md)
