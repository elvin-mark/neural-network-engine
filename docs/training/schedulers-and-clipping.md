# Schedulers & Gradient Clipping

Learning rate adjustment strategies and numerical stabilization techniques during training.

**Source files**:
- Learning Rate Schedulers: [`src/optim/scheduler.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/scheduler.rs)
- Gradient Clipping: [`src/optim/clip.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/clip.rs)
- Integration Tests: [`tests/optim_scheduler_clip_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/optim_scheduler_clip_tests.rs)

---

## The `LRScheduler` Trait

```rust
pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
}
```

---

## Supported Learning Rate Schedulers

### 1. `StepLR`
Decays learning rate by factor $\gamma$ every `step_size` epochs:
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

```rust
let mut scheduler = StepLR::new(initial_lr: 0.1, step_size: 30, gamma: 0.1);
```

### 2. `MultiStepLR`
Decays learning rate by factor $\gamma$ at predefined epoch milestones:
$$\eta_t = \eta_0 \cdot \gamma^{\sum_i \mathbb{I}(t \ge \text{milestone}_i)}$$

```rust
let mut scheduler = MultiStepLR::new(initial_lr: 0.1, milestones: vec![30, 60, 80], gamma: 0.2);
```

### 3. `ExponentialLR`
Decays learning rate exponentially each epoch:
$$\eta_t = \eta_0 \cdot \gamma^t$$

### 4. `CosineAnnealingLR`
Anneals learning rate according to a cosine curve down to $\eta_{\min}$:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$

```rust
let mut scheduler = CosineAnnealingLR::new(initial_lr: 1e-3, t_max: 100, eta_min: 1e-6);
```

### 5. `LinearWarmupCosineLR`
Linearly warms up the learning rate from $0$ to $\eta_{\max}$ for $T_{\text{warmup}}$ steps, then follows cosine decay:

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}} \pi\right)\right) & t \ge T_{\text{warmup}} \end{cases}$$

---

## Gradient Clipping

Prevents exploding gradients in recurrent models (RNNs/LSTMs) and deep transformers.

### 1. Global $L_2$ Norm Clipping (`clip_grad_norm`)
Computes the total Euclidean norm over all parameter gradients concatenated:
$$\|g\|_2 = \sqrt{\sum_{\theta \in \Theta} \sum_{i} g_{\theta, i}^2}$$
If $\|g\|_2 > \text{max\_norm}$, all gradients are scaled proportionally in-place:
$$g \leftarrow g \times \frac{\text{max\_norm}}{\|g\|_2 + 10^{-6}}$$

```rust
let total_norm = clip_grad_norm(&model.parameters(), max_norm: 1.0)?;
```

### 2. Value Clipping (`clip_grad_value`)
Clamps each gradient coordinate independently to $[-\text{clip\_value}, \text{clip\_value}]$:
$$g_i \leftarrow \max(-\text{clip\_value}, \min(\text{clip\_value}, g_i))$$

```rust
clip_grad_value(&model.parameters(), clip_value: 0.5)?;
```

---

## See Also
- [Optimizers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/optimizers.md)
- [Automatic Mixed Precision](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/mixed-precision.md)
