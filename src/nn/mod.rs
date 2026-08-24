//! Composable neural network layers, activation functions, losses, initializations, recurrent networks, quantized layers, residual networks, flash attention, and transformer modules.

pub mod activations;
pub mod attention;
pub mod bert;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod flash_attention;
pub mod init;
pub mod kv_cache;
pub mod linear;
pub mod llama;
pub mod loss;
pub mod module;
pub mod norm;
pub mod pooling;
pub mod quantized;
pub mod resnet;
pub mod rnn;
pub mod sequential;
pub mod transformer;
pub mod vit;
pub mod whisper;

pub use activations::{LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh, GELU};
pub use attention::MultiHeadAttention;
pub use bert::{
    BertConfig, BertEmbeddings, BertEncoder, BertForQuestionAnswering, BertForSequenceEmbedding,
    BertLayer, BertModel, BertPooler,
};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flash_attention::{flash_attention_forward, FlashAttention};
pub use init::{
    calculate_fan_in_and_fan_out, calculate_gain, constant, constant_, kaiming_normal,
    kaiming_normal_, kaiming_uniform, kaiming_uniform_, normal, normal_, ones_, orthogonal,
    orthogonal_, uniform, uniform_, xavier_normal, xavier_normal_, xavier_uniform, xavier_uniform_,
    zeros_, FanMode, NonLinearity,
};
pub use kv_cache::KVCache;
pub use linear::Linear;
pub use llama::{
    GroupedQueryAttention, Llama2Block, Llama2LM, LlamaConfig, RotaryEmbedding, SwiGLU,
};
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss};
pub use module::Module;
pub use norm::{BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm};
pub use pooling::MaxPool2d;
pub use quantized::{Int8Tensor, QLinear};
pub use resnet::{BottleneckBlock, ResBlock, ResNet, ResidualBlock};
pub use rnn::{GRUCell, LSTMCell, RNNActivation, RNNCell, GRU, LSTM, RNN};
pub use sequential::Sequential;
pub use transformer::{TransformerBlock, TransformerLM};
pub use vit::{ViTConfig, VisionTransformer};
pub use whisper::{Whisper, WhisperConfig, WhisperDecoder, WhisperEncoder};
