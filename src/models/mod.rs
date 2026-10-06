//! Production-ready reference neural network model architectures.
//!
//! Includes Transformer LLMs, Computer Vision architectures, Speech Recognition models,
//! and Bidirectional Encoders.

pub mod bert;
pub mod gpt2;
pub mod llama;
pub mod modern_bert;
pub mod resnet;
pub mod vit;
pub mod whisper;

pub use bert::{
    BertConfig, BertEmbeddings, BertEncoder, BertForQuestionAnswering, BertForSequenceEmbedding,
    BertLayer, BertModel, BertPooler,
};
pub use gpt2::{GPT2Block, GPT2Config, GPT2Model};
pub use llama::{
    GroupedQueryAttention, Llama2Block, Llama2LM, LlamaConfig, RotaryEmbedding, SwiGLU,
};
pub use modern_bert::{
    ModernBertAttention, ModernBertConfig, ModernBertEncoder, ModernBertForSequenceClassification,
    ModernBertForSequenceEmbedding, ModernBertLayer, ModernBertMLP, ModernBertModel,
};
pub use resnet::{BottleneckBlock, ResBlock, ResNet, ResidualBlock};
pub use vit::{ViTConfig, VisionTransformer};
pub use whisper::{Whisper, WhisperConfig, WhisperDecoder, WhisperEncoder};
