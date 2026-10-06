# Loss Functions

Numerically stabilized objective functions for classification, regression, and sequence generation.

**Source files**:
- Core implementation: [`src/nn/loss.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/nn/loss.rs)

---

## Supported Loss Functions

### 1. `CrossEntropyLoss` (Multiclass Classification)
Combines `LogSoftmax` and `NLLLoss` into a single operation using the numerically stable **Log-Sum-Exp** formulation:

$$\mathcal{L}(x, y) = -x_{y} + \log\left(\sum_{j} e^{x_j}\right) = -x_y + m + \log\left(\sum_j e^{x_j - m}\right)$$
where $m = \max_j x_j$.

- **Logits Shape**: `[Batch, NumClasses]` or `[Batch, SeqLen, VocabSize]`
- **Targets Shape**: Class indices `[Batch]` or `[Batch, SeqLen]` (as integers encoded in `f32`)
- **Backward Gradient**:
  $$\frac{\partial \mathcal{L}}{\partial x_i} = P_i - \mathbb{I}(i = y) = \text{Softmax}(x)_i - \mathbb{I}(i = y)$$

```rust
let criterion = CrossEntropyLoss::new();
let loss = criterion.forward(&logits, &targets)?;
```

### 2. `MSELoss` (Mean Squared Error)
Measures the squared Euclidean distance between predictions and ground-truth values:

$$\mathcal{L}(y_{\text{pred}}, y_{\text{true}}) = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}, i} - y_{\text{true}, i})^2$$

- **Backward Gradient**:
  $$\frac{\partial \mathcal{L}}{\partial y_{\text{pred}, i}} = \frac{2}{N} (y_{\text{pred}, i} - y_{\text{true}, i})$$

```rust
let criterion = MSELoss::new();
let loss = criterion.forward(&predictions, &targets)?;
```

### 3. `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
Combines a sigmoid activation and binary cross-entropy loss into a single operation:

$$\mathcal{L}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

By formulating the log-sigmoid term using $-|x|$, this avoids floating-point overflow for large positive or negative logits.

```rust
let criterion = BCEWithLogitsLoss::new();
let loss = criterion.forward(&logits, &binary_targets)?;
```

### 4. `L1Loss` (Mean Absolute Error)
Computes the mean absolute error:

$$\mathcal{L}(y_{\text{pred}}, y_{\text{true}}) = \frac{1}{N} \sum_{i=1}^N |y_{\text{pred}, i} - y_{\text{true}, i}|$$

- **Backward Gradient**:
  $$\frac{\partial \mathcal{L}}{\partial y_{\text{pred}, i}} = \frac{1}{N} \text{sign}(y_{\text{pred}, i} - y_{\text{true}, i})$$

---

## See Also
- [Activation Functions](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/activations.md)
- [Optimizers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/optimizers.md)
- [Automatic Differentiation Engine](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/autograd.md)
