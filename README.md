# Neural Network Engine (Pure Rust)

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An efficient, pure-Rust deep learning engine built from scratch with zero C/BLAS runtime dependencies. Features dynamic reverse-mode automatic differentiation (Autograd), cache-blocked parallel GEMM, standard neural network modules, optimizers, SafeTensors serialization, finite-difference numerical verification (`gradcheck`), and end-to-end training examples.

---

## Key Features

- **N-Dimensional Strided Tensor Runtime**:
  - Contiguous & non-contiguous strided memory views.
  - Zero-copy transposes, slicing, and reshaping.
  - NumPy/PyTorch-compliant multidirectional broadcasting.
  - Copy-on-write memory buffers (`Arc<Vec<f32>>`).

- **High-Performance Compute Engine**:
  - Cache-blocked tiled matrix multiplication ($M_C \times K_C \times N_C$) with inner-loop vectorization.
  - Multi-threaded parallelism powered by Rayon work-stealing thread pools.
  - Spatial 2D convolution lowered to GEMM via `im2col` and `col2im`.

- **Dynamic Autograd DAG (Reverse-Mode Automatic Differentiation)**:
  - Reference-counted computational graph with natural operator overloading (`+`, `-`, `*`, `/`, `neg`, `matmul`).
  - Topological sort via depth-first search for arbitrary DAG execution.
  - In-place gradient accumulation (`grad += incoming_grad`).
  - Thread-local `no_grad` guard for zero-overhead inference.

- **Modular Neural Network Primitives (`nn`)**:
  - **Layers**: `Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `BatchNorm1d`, `Dropout`, `Embedding`, `Sequential`.
  - **Activations**: `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`, `LogSoftmax`.
  - **Losses**: Numerically stable `CrossEntropyLoss` (log-sum-exp), `MSELoss`, `BCEWithLogitsLoss`, `L1Loss`.

- **State-of-the-Art Optimizers (`optim`)**:
  - `SGD` (with momentum, weight decay, and Nesterov acceleration).
  - `Adam` and `AdamW` (with decoupled weight decay and bias correction).
  - `RMSprop` (with smoothing constant and momentum).

- **Serialization & Checkpointing (`io`)**:
  - Zero-copy HuggingFace standard **SafeTensors** format (`save_safetensors`, `load_safetensors`).
  - Serde **JSON** and binary **Bincode** model & optimizer checkpointing (`Checkpoint`).

- **Mathematical Verification (`utils`)**:
  - Automated central finite-difference gradient checker (`gradcheck`) testing analytical backward gradients to $< 10^{-3}$ relative error tolerance.

---

## Directory Structure

```
neural-network-engine/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Crate root & prelude exports
│   ├── error.rs                 # Comprehensive Error and Result types
│   ├── tensor/                  # Multi-dimensional strided tensor runtime
│   │   ├── mod.rs
│   │   ├── storage.rs           # Contiguous memory buffer & copy-on-write storage
│   │   ├── shape.rs             # Shape, strides, contiguous checks, broadcasting
│   │   ├── ops.rs               # Elementwise arithmetic (+, -, *, /), pow, exp, log
│   │   ├── matmul.rs            # Tiled cache-blocked multi-threaded Rayon GEMM
│   │   ├── reduce.rs            # Sum, mean, max, argmax along axes & logsumexp
│   │   ├── conv.rs              # Im2col & col2im spatial transformations for fast Conv2d
│   │   └── display.rs           # Pretty-printing for N-dimensional tensors
│   ├── autograd/                # Dynamic reverse-mode autodiff engine
│   │   ├── mod.rs
│   │   ├── node.rs              # Graph node, tape, and backward gradient functions
│   │   ├── tensor.rs            # Autograd-enabled `Tensor` wrapper and operator overloads
│   │   └── context.rs           # Thread-local `no_grad` guard
│   ├── nn/                      # Composable deep learning layers & architectures
│   │   ├── mod.rs
│   │   ├── module.rs            # `Module` trait and parameter registration
│   │   ├── linear.rs            # Fully-connected dense layer with Kaiming/Xavier init
│   │   ├── conv.rs              # 2D Convolution layer (Conv2d)
│   │   ├── pooling.rs           # MaxPool2d and AvgPool2d
│   │   ├── norm.rs              # LayerNorm and BatchNorm1d
│   │   ├── dropout.rs           # Dropout with training/eval modes
│   │   ├── embedding.rs         # Embedding lookup table
│   │   ├── sequential.rs        # Sequential container
│   │   ├── activations.rs       # ReLU, GELU, Sigmoid, Tanh, Softmax, LeakyReLU
│   │   └── loss.rs              # MSE, CrossEntropy (log-sum-exp), BCE, L1
│   ├── optim/                   # Parameter update algorithms
│   │   ├── mod.rs
│   │   ├── sgd.rs               # SGD with Nesterov / Momentum and weight decay
│   │   ├── adam.rs              # Adam & AdamW with decoupled weight decay
│   │   └── rmsprop.rs           # RMSprop optimizer
│   ├── io/                      # Model serialization and weights I/O
│   │   ├── mod.rs
│   │   ├── safetensors.rs       # SafeTensors reader/writer for standard tensor interop
│   │   └── checkpoint.rs        # Serde-based model and optimizer state checkpointing
│   └── utils/
│       ├── mod.rs
│       ├── gradcheck.rs         # Finite-difference numerical gradient verification
│       └── data.rs              # Dataset, DataLoader, Batching, and Shuffling utilities
├── benches/
│   └── gemm_bench.rs            # Matrix multiplication & throughput benchmarks
├── tests/
│   ├── tensor_tests.rs          # Tensor math, slicing, broadcasting tests
│   ├── autograd_tests.rs        # Autograd DAG, multi-branch, cycle-free gradient tests
│   ├── nn_tests.rs              # Layers, backprop through Conv2d/Linear/Norms
│   ├── gradcheck_tests.rs       # Rigorous numerical gradient verification for all ops
│   └── serialization_tests.rs   # SafeTensors save/load round-trip tests
└── examples/
    ├── 01_spiral_mlp.rs         # Non-linear 2D spiral classification using MLP + Adam
    ├── 02_mnist_convnet.rs      # ConvNet training on image classification dataset
    ├── 03_character_lm.rs       # Character-level neural language model with Embeddings
    ├── 04_transformer_lm.rs     # nanoGPT causal decoder transformer language model
    ├── 05_llama2_gqa.rs         # LLaMA 2 with GQA, RoPE, RMSNorm & SwiGLU
    ├── 06_iris_classification.rs# Fisher's Iris dataset MLP classification & confusion matrix
    └── 07_digits_recognition.rs # 8x8 optical handwritten digits CNN recognition & ASCII viz
```

---

## Quickstart

### 1. Basic Autograd Operations

```rust
use neural_network_engine::prelude::*;

fn main() -> Result<()> {
    let a = Tensor::new(RawTensor::from_slice(&[2.0, 3.0], &[2]), true);
    let b = Tensor::new(RawTensor::from_slice(&[4.0, 5.0], &[2]), true);

    // Dynamic graph execution: c = a * b + a
    let c = a.mul(&b)?.add(&a)?;
    let loss = c.sum_all();

    // Reverse-mode backpropagation
    loss.backward();

    println!("Gradient w.r.t a: {}", a.grad().unwrap()); // [5.0, 6.0]
    println!("Gradient w.r.t b: {}", b.grad().unwrap()); // [2.0, 3.0]
    Ok(())
}
```

### 2. Building & Training a Neural Network

```rust
use neural_network_engine::prelude::*;

fn main() -> Result<()> {
    // 1. Construct model using Sequential
    let model = Sequential::new()
        .add(Linear::new(2, 32))
        .add(ReLU)
        .add(Linear::new(32, 3));

    // 2. Setup Adam optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // 3. Training step
    let x = Tensor::randn(&[64, 2], 0.0, 1.0, false);
    let labels = vec![0; 64];

    let logits = model.forward(&x)?;
    let loss = CrossEntropyLoss::forward_with_indices(&logits, &labels)?;

    optimizer.zero_grad();
    loss.backward();
    optimizer.step()?;

    println!("Step loss: {}", loss.item());
    Ok(())
}
```

---

## Running Tests, Examples & Benchmarks

### Run All Tests
```bash
cargo test
```

### Run Examples

1. **Non-linear 2D Spiral Classifier (MLP + Adam)**:
   ```bash
   cargo run --release --example 01_spiral_mlp
   ```

2. **ConvNet Image Classifier with SafeTensors Serialization**:
   ```bash
   cargo run --release --example 02_mnist_convnet
   ```

3. **Character-Level Autoregressive Language Model (Embeddings + AdamW)**:
   ```bash
   cargo run --release --example 03_character_lm
   ```

4. **Decoder-Only Causal Transformer Language Model (nanoGPT architecture)**:
   ```bash
   cargo run --release --example 04_transformer_lm
   ```

5. **LLaMA 2 with Grouped-Query Attention (GQA), RoPE, RMSNorm & SwiGLU**:
   ```bash
   cargo run --release --example 05_llama2_gqa
   ```

6. **Fisher's Iris Flower Classification (MLP + AdamW + Confusion Matrix)**:
   ```bash
   cargo run --release --example 06_iris_classification
   ```

7. **8x8 Optical Handwritten Digits Recognition (CNN + SafeTensors)**:
   ```bash
   cargo run --release --example 07_digits_recognition
   ```


### Run Benchmarks
```bash
cargo bench
```

---

## License

Dual-licensed under MIT or Apache 2.0 at your option.
