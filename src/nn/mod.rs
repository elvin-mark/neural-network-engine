//! Composable neural network layers, activation functions, losses, and transformer modules.

pub mod activations;
pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod llama;
pub mod loss;
pub mod module;
pub mod norm;
pub mod pooling;
pub mod sequential;
pub mod transformer;
pub mod vit;
pub mod whisper;

pub use activations::{LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh, GELU};
pub use attention::MultiHeadAttention;
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use llama::{
    GroupedQueryAttention, Llama2Block, Llama2LM, LlamaConfig, RotaryEmbedding, SwiGLU,
};
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss};
pub use module::Module;
pub use norm::{BatchNorm1d, LayerNorm, RMSNorm};
pub use pooling::MaxPool2d;
pub use sequential::Sequential;
pub use transformer::{TransformerBlock, TransformerLM};
pub use vit::{ViTConfig, VisionTransformer};
pub use whisper::{Whisper, WhisperConfig, WhisperDecoder, WhisperEncoder};
