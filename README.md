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
  - **Zero-Allocation Memory Pool (`TensorPool`)**: Thread-local power-of-two binned memory recycler achieving **>98% cache hit rates**, eliminating OS heap allocation thrashing during training loops.

- **Modular Neural Network Primitives (`nn`)**:
  - **Layers**: `Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `BatchNorm2d`, `Dropout`, `Embedding`, `Sequential`.
  - **Activations**: `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`, `LogSoftmax`.
  - **Losses**: Numerically stable `CrossEntropyLoss` (log-sum-exp), `MSELoss`, `BCEWithLogitsLoss`, `L1Loss`.

- **Residual Vision Networks (`nn::resnet`)**:
  - **Skip Connections**: Basic `ResidualBlock` (2x $3\times3$ Conv2D) and `BottleneckBlock` ($1\times1 \to 3\times3 \to 1\times1$ Conv2D with 4x channel expansion).
  - **Architectures**: Configurable `ResNet-18`, `ResNet-34`, `ResNet-50` with support for both standard ImageNet and $32\times32$ CIFAR stems.

- **Computer Vision Data Augmentations (`vision::transforms`)**:
  - **Composable Pipeline (`Compose`)**: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCrop` (with spatial padding), `Normalize` (ImageNet, CIFAR-10, CIFAR-100 presets), `ColorJitter`, and `RandomRotation90`.

- **Inference Acceleration, Attention & Low-Precision Quantization (`nn::kv_cache`, `nn::flash_attention`, `nn::moe`, `nn::quantized`)**:
  - **Mixture of Experts (`MoELayer`, `SparseMoEBlock`, `TopKRouter`)**: Sparse Top-K gating (Mixtral 8x7B / DeepSeek-V2 style) routing tokens to subsets of expert FFNs with auxiliary load-balancing loss.
  - **FlashAttention-2 (`FlashAttention`)**: Tiled online-softmax attention operating within L1/L2 CPU cache blocks, reducing attention memory from **$O(T^2) \to O(T)$** (256x memory reduction and 3.1x speedup).
  - **$O(N)$ Key-Value Cache (`KVCache`)**: Eliminates quadratic attention recomputation during autoregressive generation in LLaMA 2 and Transformers (3.7x+ speedup).
  - **INT8 Weight Quantization (`QLinear`, `Int8Tensor`)**: Symmetric per-channel weight compression achieving 4x memory reduction (75% savings) and 3.1x faster GEMM with AVX2 SIMD dot-product kernels.

- **Recurrent Neural Networks (`nn::rnn`)**:
  - **Elman RNN**: Single-step `RNNCell` and multi-layer sequence `RNN` with Tanh / ReLU non-linearities and bidirectional support.
  - **LSTM (Long Short-Term Memory)**: `LSTMCell` and multi-layer sequence `LSTM` with fused gates ($i, f, g, o$) and forget gate bias initialization.
  - **GRU (Gated Recurrent Unit)**: `GRUCell` and multi-layer sequence `GRU` with reset, update, and candidate gates ($r, z, n$).

- **Training Stability, Schedulers & Gradient Utilities (`optim`)**:
  - **Automatic Mixed Precision (`LossScaler`)**: Dynamic loss scaling with NaN/Inf detection, gradient unscaling, and adaptive backoff/growth factors.
  - **Gradient Clipping**: `clip_grad_norm` ($L_2$ global norm clipping across all parameters) and `clip_grad_value` (element-wise bounding).
  - **Learning Rate Schedulers**: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, and `LinearWarmupCosineLR`.
  - **Weight Initialization (`nn::init`)**: `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `orthogonal` (Gram-Schmidt QR decomposition), and activation gain calculators.

- **Python Bindings & NumPy Interoperability (`pyo3`, `numpy`)**:
  - Full Python package with zero-copy NumPy array bridging (`Tensor.from_numpy`, `tensor.to_numpy()`).
  - Python classes for `Tensor` (operator overloads, backward, gradients), `Linear`, `LayerNorm`, `RMSNorm`, activations, `SGD`, `Adam`, and `LossScaler`.
  - Seamless integration via `pyproject.toml` and Maturin.

- **State-of-the-Art Optimizers (`optim`)**:
  - `SGD` (with momentum, weight decay, and Nesterov acceleration).
  - `Adam` and `AdamW` (with decoupled weight decay and bias correction).
  - `RMSprop` (with smoothing constant and momentum).

- **Serialization & Checkpointing (`io`)**:
  - Zero-copy HuggingFace standard **SafeTensors** format (`save_safetensors`, `load_safetensors`).
  - Serde **JSON** and binary **Bincode** model & optimizer checkpointing (`Checkpoint`).

- **Byte-Level BPE Tokenizer (`tokenizer`)**:
  - Pure-Rust Byte-Level Byte-Pair Encoding (BPE) with 100% UTF-8 coverage and zero out-of-vocabulary errors.
  - Word/whitespace pre-tokenization boundary preservation and configurable special tokens (`<s>`, `</s>`, `<unk>`, `<pad>`).
  - JSON serialization and deserialization (`save_json`, `load_json`).

- **Audio Processing & Speech Recognition (`nn::whisper`, `utils::audio`)**:
  - **OpenAI Whisper Architecture**: 1D Conv audio downsampling, bidirectional Transformer encoder, and Transformer decoder with causal self-attention and cross-attention.
  - **Audio Frontend**: Triangular Mel filterbanks and STFT Log-Mel Spectrogram computation from raw audio waveforms.
  - **Autoregressive Decoding**: Greedy and sampling-based speech-to-text transcription.

- **Bidirectional Transformers & Question Answering (`nn::bert`)**:
  - **BERT Architecture**: Word + 1D Position + Segment/Token-Type Tri-Embeddings, bidirectional Multi-Head Self-Attention layers, and `[CLS]` sentence pooler.
  - **Task Heads**: Span extraction head for Extractive Question Answering (`BertForQuestionAnswering`) and normalized sentence embedding head with cosine similarity ranking (`BertForSequenceEmbedding`).

- **Hardware GPU Acceleration (`gpu`, `wgpu`)**:
  - **WebGPU Compute Engine**: Pure-Rust GPU compute backend compiling WGSL shaders to Vulkan (Linux), Metal (macOS), and DirectX 12 (Windows).
  - **GPU Primitives (`GpuTensor`)**: 16x16 shared-memory tiled GEMM, parallel elementwise arithmetic & activations, row-wise Softmax, and LayerNorm/RMSNorm reduction kernels in VRAM.
  - **Zero-Copy Pipeline**: Model weights and activations execute 100% inside GPU VRAM with zero PCIe roundtrips during multi-layer forward passes.

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

### Downloading Real-World Datasets (Optional)
To download standard benchmark datasets (Fisher's Iris, 8x8 Digits, MNIST, CIFAR-10, and CIFAR-100) into `data/` (gitignored):
```bash
./scripts/download_datasets.sh
# or using Python:
python3 scripts/download_datasets.py
```
*(If the `data/` directory is not present, all examples will seamlessly use high-fidelity built-in synthetic datasets).*

### Run All Tests
```bash
cargo test
```

### Run Examples

1. **Non-linear 2D Spiral Classifier (MLP + Adam)**:
   ```bash
   cargo run --release --example 01_spiral_mlp
   ```

2. **28x28 MNIST Handwritten Digits CNN Classifier (SafeTensors)**:
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

8. **32x32 RGB CIFAR-10 10-Class ConvNet Image Classifier**:
   ```bash
   cargo run --release --example 08_cifar10_convnet
   ```

9. **32x32 RGB CIFAR-100 100-Class ConvNet with Top-1 / Top-5 Evaluation**:
   ```bash
   cargo run --release --example 09_cifar100_convnet
   ```

10. **Vision Transformer (ViT) on 32x32 RGB CIFAR-100 (Patch Embedding + Multi-Head Attention)**:
    ```bash
    cargo run --release --example 10_cifar100_vit
    ```

11. **Byte-Level BPE Tokenizer + LLaMA 2 Language Model on TinyStories**:
    ```bash
    cargo run --release --example 11_llama_bpe_training
    ```

12. **Whisper Sequence-to-Sequence Speech Recognition on Spoken Audio**:
    ```bash
    cargo run --release --example 12_whisper_speech_recognition
    ```

13. **BERT Extractive Question Answering & Semantic Text Embeddings**:
    ```bash
    cargo run --release --example 13_bert_qa_embeddings
    ```

14. **Hardware WebGPU Compute Acceleration (Matrix Multiplication & Deep NN Forward Pass in VRAM)**:
    ```bash
    cargo run --release --features gpu --example 14_gpu_acceleration
    ```

15. **Recurrent Sequence Modeling & Benchmark (Elman RNN vs LSTM vs GRU)**:
    ```bash
    cargo run --release --example 15_recurrent_sequence_models
    ```

16. **Key-Value Cache (KV-Cache) & INT8 Quantization (QLinear) Benchmark**:
    ```bash
    cargo run --release --example 16_kvcache_and_int8_quantization
    ```

17. **Deep Residual Networks (ResNet-18) & Vision Data Augmentation Pipeline**:
    ```bash
    cargo run --release --example 17_resnet_cifar_vision
    ```

18. **Zero-Allocation Tensor Pool (`TensorPool`) & FlashAttention-2 Benchmark**:
    ```bash
    cargo run --release --example 18_tensor_pool_and_flash_attention
    ```

19. **Mixture of Experts (MoE) & Automatic Mixed Precision (AMP) Benchmark**:
    ```bash
    cargo run --release --example 19_moe_and_mixed_precision
    ```

### Python Examples & Workflows
All Python examples live in `python/examples/` and utilize the compiled pure-Rust backend:

```bash
# Build python extension library
cargo build --release --features python

# 1. End-to-End NumPy MLP Training with AMP LossScaler
python3 python/examples/training_demo.py

# 2. ResNet-18 Vision Training with BatchNorm2d & CrossEntropyLoss
python3 python/examples/resnet_vision.py

# 3. Transformer Language Model (nanoGPT style) with Causal Self-Attention
python3 python/examples/transformer_lm.py

# 4. Mixture of Experts (MoE) & FlashAttention-2 vs Standard Attention
python3 python/examples/moe_and_flash_attention.py
```

### Run Benchmarks
```bash
cargo bench
```

### Build 100% Pure Static Binaries (Zero Dynamic Dependencies)
To produce a completely self-contained binary with **zero `.so` dependencies** (e.g. for `FROM scratch` Docker containers or Alpine Linux):

```bash
# 1. Add musl target
rustup target add x86_64-unknown-linux-musl

# 2. Build statically linked release binary
cargo build --release --target x86_64-unknown-linux-musl --example 13_bert_qa_embeddings

# 3. Verify static linkage (0 shared libraries linked)
file target/x86_64-unknown-linux-musl/release/examples/13_bert_qa_embeddings
# Output: ELF 64-bit LSB pie executable, x86-64, statically linked
```

---

## License

Dual-licensed under MIT or Apache 2.0 at your option.
