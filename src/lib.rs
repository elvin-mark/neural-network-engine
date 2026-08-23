//! # Neural Network Engine
//!
//! An efficient, pure-Rust deep learning engine featuring:
//! - Multi-dimensional strided tensor runtime with broadcasting and cache-blocked SIMD/Rayon GEMM
//! - Dynamic reverse-mode automatic differentiation (Autograd) DAG with in-place gradient accumulation
//! - Composable deep learning layers (`Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `Dropout`, `Embedding`, `Sequential`, `RNN`, `LSTM`, `GRU`, `MultiHeadAttention`, `GroupedQueryAttention`, `TransformerBlock`, `Llama2Block`, `TransformerLM`, `Llama2LM`)
//! - Recurrent layers: Elman RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) with bidirectional and multi-layer sequence support
//! - Modern LLM primitives: Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU Feed-Forward Networks
//! - Numerically stable loss functions (`CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`, `L1Loss`)
//! - Optimizers (`SGD`, `Adam`, `AdamW`, `RMSprop`), gradient clipping, and learning rate schedulers
//! - Mathematically principled weight initializations (Xavier, Kaiming, Orthogonal)
//! - SafeTensors and JSON/Bincode serialization
//! - Numerical finite-difference gradient verification (`gradcheck`)

pub mod autograd;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod tokenizer;
pub mod utils;

pub use autograd::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard, Tensor};
pub use error::{EngineError, Result};
#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuLayerNorm, GpuLinear, GpuRMSNorm, GpuTensor, ToGpu};
pub use tensor::RawTensor;
pub use tokenizer::ByteLevelBPE;

/// Commonly used imports grouped for convenience.
pub mod prelude {
    pub use crate::autograd::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard, Tensor};
    pub use crate::error::{EngineError, Result};
    #[cfg(feature = "gpu")]
    pub use crate::gpu::{GpuContext, GpuLayerNorm, GpuLinear, GpuRMSNorm, GpuTensor, ToGpu};
    pub use crate::io::{load_safetensors, save_safetensors, Checkpoint};
    pub use crate::nn::{
        calculate_fan_in_and_fan_out, calculate_gain, constant, constant_, kaiming_normal,
        kaiming_normal_, kaiming_uniform, kaiming_uniform_, normal, normal_, ones_, orthogonal,
        orthogonal_, uniform, uniform_, xavier_normal, xavier_normal_, xavier_uniform,
        xavier_uniform_, zeros_, BCEWithLogitsLoss, BatchNorm1d, BertConfig, BertEmbeddings,
        BertEncoder, BertForQuestionAnswering, BertForSequenceEmbedding, BertLayer, BertModel,
        BertPooler, Conv2d, CrossEntropyLoss, Dropout, Embedding, FanMode, GRUCell,
        GroupedQueryAttention, L1Loss, LSTMCell, LayerNorm, LeakyReLU, Linear, Llama2Block,
        Llama2LM, LlamaConfig, MSELoss, MaxPool2d, Module, MultiHeadAttention, NonLinearity,
        RMSNorm, RNNActivation, RNNCell, ReLU, RotaryEmbedding, Sequential, SiLU, Sigmoid, Softmax,
        SwiGLU, Tanh, TransformerBlock, TransformerLM, ViTConfig, VisionTransformer, Whisper,
        WhisperConfig, GELU, GRU, LSTM, RNN,
    };
    pub use crate::optim::{
        clip_grad_norm, clip_grad_value, Adam, CosineAnnealingLR, ExponentialLR, LRScheduler,
        LinearWarmupCosineLR, MultiStepLR, Optimizer, RMSprop, StepLR, SGD,
    };
    pub use crate::tensor::conv::Conv2dParams;
    pub use crate::tensor::RawTensor;
    pub use crate::tokenizer::ByteLevelBPE;
    pub use crate::utils::{
        compute_log_mel_spectrogram, create_mel_filterbank, generate_cifar100_dataset,
        generate_cifar10_dataset, generate_digits_dataset, generate_mnist_dataset,
        generate_qa_dataset, generate_semantic_similarity_dataset, generate_spiral_dataset,
        generate_spoken_dataset, generate_tinystories_dataset, generate_xor_dataset, gradcheck,
        hz_to_mel, load_cifar100_dataset, load_cifar100_from_binary, load_cifar10_dataset,
        load_cifar10_from_binary, load_digits_dataset, load_digits_from_csv, load_iris_dataset,
        load_iris_from_csv, load_mnist_dataset, load_mnist_from_idx, load_spoken_dataset,
        load_tinystories_dataset, mel_to_hz, standardize, synthesize_spoken_word, train_test_split,
        DataLoader, QASample, TensorDataset, CIFAR10_CLASSES, SPOKEN_CLASSES,
    };
}
