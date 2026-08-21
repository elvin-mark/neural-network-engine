//! # Neural Network Engine
//!
//! An efficient, pure-Rust deep learning engine featuring:
//! - Multi-dimensional strided tensor runtime with broadcasting and cache-blocked SIMD/Rayon GEMM
//! - Dynamic reverse-mode automatic differentiation (Autograd) DAG with in-place gradient accumulation
//! - Composable deep learning layers (`Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `BatchNorm1d`, `Dropout`, `Embedding`, `Sequential`, `MultiHeadAttention`, `TransformerBlock`, `TransformerLM`)
//! - Numerically stable loss functions (`CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`, `L1Loss`)
//! - Optimizers (`SGD`, `Adam`, `AdamW`, `RMSprop`)
//! - SafeTensors and JSON/Bincode serialization
//! - Numerical finite-difference gradient verification (`gradcheck`)

pub mod autograd;
pub mod error;
pub mod io;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod utils;

pub use autograd::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard, Tensor};
pub use error::{EngineError, Result};
pub use tensor::RawTensor;

/// Commonly used imports grouped for convenience.
pub mod prelude {
    pub use crate::autograd::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard, Tensor};
    pub use crate::error::{EngineError, Result};
    pub use crate::io::{load_safetensors, save_safetensors, Checkpoint};
    pub use crate::nn::{
        BCEWithLogitsLoss, BatchNorm1d, Conv2d, CrossEntropyLoss, Dropout, Embedding, L1Loss,
        LayerNorm, LeakyReLU, Linear, MSELoss, MaxPool2d, Module, MultiHeadAttention, ReLU,
        Sequential, Sigmoid, Softmax, Tanh, TransformerBlock, TransformerLM, GELU,
    };
    pub use crate::optim::{Adam, RMSprop, SGD};
    pub use crate::tensor::conv::Conv2dParams;
    pub use crate::tensor::RawTensor;
    pub use crate::utils::{
        generate_spiral_dataset, generate_xor_dataset, gradcheck, DataLoader, TensorDataset,
    };
}
