# Optimizers (`optim::Optimizer`)

State-of-the-art first-order optimization algorithms for parameter updates during backpropagation.

**Source files**:
- Optimizer Trait: [`src/optim/mod.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/mod.rs)
- SGD: [`src/optim/sgd.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/sgd.rs)
- Adam & AdamW: [`src/optim/adam.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/adam.rs)
- RMSprop: [`src/optim/rmsprop.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/optim/rmsprop.rs)

---

## The `Optimizer` Trait

```rust
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}
```

---

## Supported Optimizers

### 1. `SGD` (Stochastic Gradient Descent)
Supports classical momentum, Nesterov accelerated momentum, and $L_2$ weight decay:

$$v_t = \mu \cdot v_{t-1} + (g_t + \lambda \theta_{t-1})$$
$$\theta_t = \begin{cases} \theta_{t-1} - \gamma v_t & \text{standard momentum} \\ \theta_{t-1} - \gamma (g_t + \lambda \theta_{t-1} + \mu v_t) & \text{Nesterov momentum} \end{cases}$$

- **Defaults**: `lr = 0.01`, `momentum = 0.0`, `weight_decay = 0.0`, `nesterov = false`

```rust
let opt = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)
    .weight_decay(1e-4)
    .nesterov(true);
```

### 2. `Adam` & `AdamW` (Adaptive Moment Estimation)
Maintains first ($m_t$) and second ($v_t$) uncentered moment estimates with bias corrections:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

#### Adam vs AdamW (Decoupled Weight Decay)
- **Standard Adam**: Applies weight decay directly to gradients: $g_t \leftarrow g_t + \lambda \theta_{t-1}$, which couples weight decay to the adaptive learning rate $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$.
- **AdamW**: Decouples weight decay from the gradient moments:
  $$\theta_t = \theta_{t-1} - \gamma \lambda \theta_{t-1} - \frac{\gamma \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- **Defaults**: `lr = 1e-3`, `betas = (0.9, 0.999)`, `eps = 1e-8`, `weight_decay = 0.01` (AdamW)

```rust
let mut opt = Adam::new(model.parameters(), 1e-3)
    .betas(0.9, 0.999)
    .eps(1e-8)
    .weight_decay(0.01)
    .adamw(true); // Toggles decoupled weight decay (AdamW)
```

### 3. `RMSprop`
Divides gradient by a running root-mean-square average of recent magnitudes:

$$v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\gamma g_t}{\sqrt{v_t} + \epsilon}$$

- **Defaults**: `lr = 1e-2`, `alpha = 0.99`, `eps = 1e-8`, `weight_decay = 0.0`

---

## Canonical Training Loop Example

```rust
let mut optimizer = Adam::new(model.parameters(), 1e-3).adamw(true);
let mut scheduler = CosineAnnealingLR::new(&optimizer, 100, 1e-5);
let criterion = CrossEntropyLoss::new();

for epoch in 0..num_epochs {
    for (inputs, targets) in dataloader.iter() {
        // 1. Zero out previous gradients
        optimizer.zero_grad();

        // 2. Forward pass
        let outputs = model.forward(&inputs)?;
        let loss = criterion.forward(&outputs, &targets)?;

        // 3. Backward pass (Autograd DAG traversal)
        loss.backward()?;

        // 4. Gradient clipping (prevents exploding gradients)
        clip_grad_norm(&model.parameters(), 1.0)?;

        // 5. Parameter update
        optimizer.step()?;
    }

    // 6. Learning rate schedule adjustment
    scheduler.step(&mut optimizer);
}
```

---

## See Also
- [Schedulers & Gradient Clipping](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/schedulers-and-clipping.md)
- [Automatic Differentiation Engine](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/autograd.md)
- [Loss Functions](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/losses.md)
