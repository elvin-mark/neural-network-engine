//! Production-ready reference neural network model architectures.
//!
//! Includes Transformer LLMs, Computer Vision architectures, Speech Recognition models,
//! and Bidirectional Encoders.

pub mod bert;
pub mod gpt2;
pub mod lfm3;
pub mod llama;
pub mod modern_bert;
pub mod qwen3_5;
pub mod resnet;
pub mod vit;
pub mod whisper;

pub use bert::{
    BertConfig, BertEmbeddings, BertEncoder, BertForQuestionAnswering, BertForSequenceEmbedding,
    BertLayer, BertModel, BertPooler,
};
pub use gpt2::{GPT2Block, GPT2Config, GPT2Model};
pub use lfm3::{
    Lfm3Attention, Lfm3Cache, Lfm3Config, Lfm3DecoderLayer, Lfm3LayerCache, Lfm3LayerType, Lfm3MLP,
    Lfm3Model, Lfm3ShortConv, Lfm3TokenMixer,
};
pub use llama::{
    GroupedQueryAttention, Llama2Block, Llama2LM, LlamaConfig, RotaryEmbedding, SwiGLU,
};
pub use modern_bert::{
    ModernBertAttention, ModernBertConfig, ModernBertEncoder, ModernBertForSequenceClassification,
    ModernBertForSequenceEmbedding, ModernBertLayer, ModernBertMLP, ModernBertModel,
};
pub use qwen3_5::{
    Qwen3_5Attention, Qwen3_5Cache, Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet,
    Qwen3_5LayerCache, Qwen3_5LayerType, Qwen3_5MLP, Qwen3_5Model, Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated, Qwen3_5TokenMixer,
};
pub use resnet::{BottleneckBlock, ResBlock, ResNet, ResidualBlock};
pub use vit::{ViTConfig, VisionTransformer};
pub use whisper::{Whisper, WhisperConfig, WhisperDecoder, WhisperEncoder, WhisperKVCache};
