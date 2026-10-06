//! OpenAI Whisper Encoder-Decoder Sequence-to-Sequence Architecture for Speech Recognition.
//!
//! Features:
//! - 1D Convolutional temporal downsampling of log-mel spectrograms
//! - Transformer Encoder with bidirectional multi-head self-attention
//! - Transformer Decoder with causal self-attention and encoder-decoder cross-attention
//! - Pre-LayerNorm residual connections and GELU feed-forward networks
//! - Autoregressive greedy speech transcription decoding

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::attention::MultiHeadAttention;
use crate::nn::conv::Conv2d;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;
use crate::tensor::RawTensor;
use crate::tokenizer::ByteLevelBPE;

/// Configuration hyperparameters for the Whisper Speech-to-Text Model.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of acoustic Mel frequency bins in input spectrograms (e.g. 64 or 80).
    pub n_mels: usize,
    /// Hidden dimension of transformer token and audio representations.
    pub d_model: usize,
    /// Number of encoder transformer layers.
    pub encoder_layers: usize,
    /// Number of decoder transformer layers.
    pub decoder_layers: usize,
    /// Number of attention heads in encoder self-attention.
    pub encoder_heads: usize,
    /// Number of attention heads in decoder self- and cross-attention.
    pub decoder_heads: usize,
    /// Inner dimension of the Feed-Forward (FFN) blocks.
    pub d_ff: usize,
    /// Target text vocabulary size.
    pub vocab_size: usize,
    /// Maximum number of encoder audio frames after convolutional downsampling.
    pub max_source_positions: usize,
    /// Maximum number of target text tokens.
    pub max_target_positions: usize,
}

impl WhisperConfig {
    /// Pre-configured compact Whisper model for fast training and spoken word recognition.
    pub fn tiny() -> Self {
        Self {
            n_mels: 64,
            d_model: 64,
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_heads: 4,
            decoder_heads: 4,
            d_ff: 160,
            vocab_size: 350,
            max_source_positions: 128,
            max_target_positions: 48,
        }
    }
}

/// A single Transformer Encoder Block for Whisper.
pub struct WhisperEncoderBlock {
    pub self_attn_ln: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub mlp_ln: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}

impl WhisperEncoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn_ln: LayerNorm::new(d_model),
            self_attn: MultiHeadAttention::new(d_model, num_heads, false),
            mlp_ln: LayerNorm::new(d_model),
            mlp_fc1: Linear::new(d_model, d_ff),
            mlp_fc2: Linear::new(d_ff, d_model),
        }
    }

    pub fn forward_block(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Bidirectional Self-Attention with Pre-LayerNorm & Residual
        let norm_x = self.self_attn_ln.forward(x)?;
        let h_attn = self.self_attn.forward_attention(&norm_x)?;
        let x = x.add(&h_attn)?;

        // 2. GELU MLP with Pre-LayerNorm & Residual
        let norm_x2 = self.mlp_ln.forward(&x)?;
        let h_mlp = self.mlp_fc1.forward(&norm_x2)?.gelu()?;
        let h_mlp = self.mlp_fc2.forward(&h_mlp)?;
        x.add(&h_mlp)
    }
}

impl Module for WhisperEncoderBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn_ln.parameters());
        params.extend(self.self_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// A single Transformer Decoder Block with Causal Self-Attention and Encoder Cross-Attention.
pub struct WhisperDecoderBlock {
    pub self_attn_ln: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub mlp_ln: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}

impl WhisperDecoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn_ln: LayerNorm::new(d_model),
            self_attn: MultiHeadAttention::new(d_model, num_heads, true), // Causal
            cross_attn_ln: LayerNorm::new(d_model),
            cross_attn: MultiHeadAttention::new(d_model, num_heads, false), // Cross-attention
            mlp_ln: LayerNorm::new(d_model),
            mlp_fc1: Linear::new(d_model, d_ff),
            mlp_fc2: Linear::new(d_ff, d_model),
        }
    }

    pub fn forward_block(&self, x: &Tensor, memory: &Tensor) -> Result<Tensor> {
        // 1. Masked Causal Self-Attention with Pre-LayerNorm & Residual
        let norm_x = self.self_attn_ln.forward(x)?;
        let h_self = self.self_attn.forward_attention(&norm_x)?;
        let x = x.add(&h_self)?;

        // 2. Multi-Head Cross-Attention with Pre-LayerNorm & Residual
        let norm_x2 = self.cross_attn_ln.forward(&x)?;
        let h_cross = self.cross_attn.forward_cross_attention(&norm_x2, memory)?;
        let x = x.add(&h_cross)?;

        // 3. GELU MLP with Pre-LayerNorm & Residual
        let norm_x3 = self.mlp_ln.forward(&x)?;
        let h_mlp = self.mlp_fc1.forward(&norm_x3)?.gelu()?;
        let h_mlp = self.mlp_fc2.forward(&h_mlp)?;
        x.add(&h_mlp)
    }
}

impl Module for WhisperDecoderBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn_ln.parameters());
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn_ln.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// The Whisper Audio Transformer Encoder.
pub struct WhisperEncoder {
    pub conv1: Conv2d,
    pub pos_embed: Tensor,
    pub blocks: Vec<WhisperEncoderBlock>,
    pub ln_post: LayerNorm,
    pub config: WhisperConfig,
}

impl WhisperEncoder {
    pub fn new(config: &WhisperConfig) -> Self {
        // Conv downsampling: kernel (n_mels, 3), stride (1, 2), padding (0, 1)
        let conv1 = Conv2d::with_options(
            1,
            config.d_model,
            (config.n_mels, 3),
            (1, 2),
            (0, 1),
            (1, 1),
            true,
        );

        let pos_data =
            RawTensor::randn(&[1, config.max_source_positions, config.d_model], 0.0, 0.02);
        let pos_embed = Tensor::new(pos_data, true);

        let mut blocks = Vec::with_capacity(config.encoder_layers);
        for _ in 0..config.encoder_layers {
            blocks.push(WhisperEncoderBlock::new(
                config.d_model,
                config.encoder_heads,
                config.d_ff,
            ));
        }

        Self {
            conv1,
            pos_embed,
            blocks,
            ln_post: LayerNorm::new(config.d_model),
            config: config.clone(),
        }
    }

    /// Encodes a batch of log-mel spectrograms [B, n_mels, T_audio] into encoder memory [B, T_enc, d_model].
    pub fn forward_encoder(&self, mel: &Tensor) -> Result<Tensor> {
        let shape = mel.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "WhisperEncoder (expected 3D mel input [B, n_mels, T])",
                shapes: vec![shape],
            });
        }

        let (b, n_mels, t) = (shape[0], shape[1], shape[2]);
        if n_mels != self.config.n_mels {
            return Err(EngineError::ShapeMismatch {
                expected: vec![b, self.config.n_mels, t],
                actual: shape,
            });
        }

        // 1. Reshape to 4D [B, 1, n_mels, T] for 2D convolution
        let x_4d = mel.reshape(&[b, 1, n_mels, t])?;

        // 2. Conv downsampling -> [B, d_model, 1, T_enc]
        let conv_out = self.conv1.forward(&x_4d)?.gelu()?;
        let t_enc = conv_out.shape()[3];

        // 3. Reshape & transpose to [B, T_enc, d_model]
        let mut x = conv_out
            .reshape(&[b, self.config.d_model, t_enc])?
            .transpose(1, 2)?;

        // 4. Add positional embeddings (truncated to t_enc)
        let pos = self.pos_embed.slice(1, 0, t_enc)?;
        x = x.add(&pos)?;

        // 5. Cascade through Encoder Blocks
        for block in &self.blocks {
            x = block.forward_block(&x)?;
        }

        // 6. Post LayerNorm
        self.ln_post.forward(&x)
    }
}

impl Module for WhisperEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_encoder(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_post.parameters());
        params
    }
}

/// The Whisper Text Transformer Decoder.
pub struct WhisperDecoder {
    pub token_embedding: Embedding,
    pub pos_embed: Tensor,
    pub blocks: Vec<WhisperDecoderBlock>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub config: WhisperConfig,
}

impl WhisperDecoder {
    pub fn new(config: &WhisperConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.d_model);

        let pos_data =
            RawTensor::randn(&[1, config.max_target_positions, config.d_model], 0.0, 0.02);
        let pos_embed = Tensor::new(pos_data, true);

        let mut blocks = Vec::with_capacity(config.decoder_layers);
        for _ in 0..config.decoder_layers {
            blocks.push(WhisperDecoderBlock::new(
                config.d_model,
                config.decoder_heads,
                config.d_ff,
            ));
        }

        Self {
            token_embedding,
            pos_embed,
            blocks,
            ln_post: LayerNorm::new(config.d_model),
            lm_head: Linear::without_bias(config.d_model, config.vocab_size),
            config: config.clone(),
        }
    }

    /// Decodes target token sequence [B, T_text] given encoder memory [B, T_enc, d_model].
    pub fn forward_decoder(&self, tokens: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let shape = tokens.shape();
        if shape.len() != 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "WhisperDecoder (expected 2D tokens input [B, T])",
                shapes: vec![shape],
            });
        }

        let (_b, t_text) = (shape[0], shape[1]);

        // 1. Token Embeddings -> [B, T_text, d_model]
        let tok_embeds = self.token_embedding.forward(tokens)?;

        // 2. Positional Embeddings
        let pos = self.pos_embed.slice(1, 0, t_text)?;
        let mut x = tok_embeds.add(&pos)?;

        // 3. Cascade through Decoder Blocks with Cross-Attention
        for block in &self.blocks {
            x = block.forward_block(&x, memory)?;
        }

        // 4. Post LayerNorm & Projection Head -> [B, T_text, vocab_size]
        let x = self.ln_post.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

impl Module for WhisperDecoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_decoder(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_post.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}

/// The complete Whisper Encoder-Decoder Model for Automatic Speech Recognition.
pub struct Whisper {
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub config: WhisperConfig,
}

impl Whisper {
    pub fn new(config: WhisperConfig) -> Self {
        let encoder = WhisperEncoder::new(&config);
        let decoder = WhisperDecoder::new(&config);
        Self {
            encoder,
            decoder,
            config,
        }
    }

    /// Forward pass: encodes audio spectrogram and decodes text tokens to output vocabulary logits.
    pub fn forward_model(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let memory = self.encoder.forward_encoder(mel)?;
        self.decoder.forward_decoder(tokens, &memory)
    }

    /// Encodes acoustic spectrogram into encoder memory representations.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward_encoder(mel)
    }

    /// Decodes text tokens using pre-computed encoder memory representations.
    pub fn decode(&self, tokens: &Tensor, memory: &Tensor) -> Result<Tensor> {
        self.decoder.forward_decoder(tokens, memory)
    }

    /// Autoregressively transcribes an audio spectrogram into text using greedy decoding.
    pub fn generate_transcription(
        &self,
        mel: &Tensor,
        tokenizer: &ByteLevelBPE,
        max_tokens: usize,
    ) -> Result<String> {
        let memory = self.encode(mel)?;

        let bos_id = tokenizer.bos_token_id().unwrap_or(1);
        let eos_id = tokenizer.eos_token_id().unwrap_or(2);

        let mut token_ids = vec![bos_id];

        for _ in 0..max_tokens {
            let cur_len = token_ids.len();
            let tokens_raw = RawTensor::from_vec(
                token_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            );
            let tokens_tensor = Tensor::new(tokens_raw, false);

            let logits = self.decode(&tokens_tensor, &memory)?;
            let slice = logits.data().to_contiguous();
            let num_classes = self.config.vocab_size;
            let last_logits = &slice.as_slice()[(cur_len - 1) * num_classes..cur_len * num_classes];

            // Greedy argmax selection
            let mut best_token = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in last_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_token = idx;
                }
            }

            if best_token == eos_id {
                break;
            }

            token_ids.push(best_token);
        }

        // Decode generated tokens (skip initial BOS token)
        let generated_slice = if token_ids.len() > 1 {
            &token_ids[1..]
        } else {
            &token_ids[..]
        };

        tokenizer.decode(generated_slice)
    }
}

impl Module for Whisper {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_model(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_forward_and_backward() {
        let config = WhisperConfig {
            n_mels: 32,
            d_model: 32,
            encoder_layers: 1,
            decoder_layers: 1,
            encoder_heads: 2,
            decoder_heads: 2,
            d_ff: 64,
            vocab_size: 100,
            max_source_positions: 64,
            max_target_positions: 32,
        };

        let whisper = Whisper::new(config);

        // Input mel: [Batch=2, n_mels=32, T_audio=20]
        let mel = Tensor::randn(&[2, 32, 20], 0.0, 1.0, true);
        // Input text tokens: [Batch=2, T_text=5]
        let tokens_raw = RawTensor::from_slice(
            &[1.0, 10.0, 12.0, 15.0, 2.0, 1.0, 5.0, 8.0, 9.0, 2.0],
            &[2, 5],
        );
        let tokens = Tensor::new(tokens_raw, false);

        let logits = whisper.forward_model(&mel, &tokens).unwrap();
        assert_eq!(logits.shape(), &[2, 5, 100]);

        // Autograd backward verification
        let loss = logits.sum_all();
        loss.backward();

        assert!(mel.grad().is_some());
        assert_eq!(mel.grad().unwrap().shape(), &[2, 32, 20]);
        assert!(whisper.encoder.conv1.weight.grad().is_some());
        assert!(whisper.decoder.lm_head.weight.grad().is_some());
    }
}
