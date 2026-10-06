//! Bidirectional Encoder Representations from Transformers (BERT).
//!
//! Features:
//! - Tri-Embedding Layer (Word Tokens + Learned 1D Positions + Sentence Token Types/Segments)
//! - Stack of Bidirectional Transformer Encoder Layers with Multi-Head Self-Attention and GELU FFNs
//! - Pooler Layer extracting dense sentence representations from the `[CLS]` token
//! - Question Answering Head (`BertForQuestionAnswering`) predicting start & end span logits
//! - Sequence Embedding Head (`BertForSequenceEmbedding`) for semantic text search and similarity

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::attention::MultiHeadAttention;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;
use crate::tensor::RawTensor;

/// Configuration hyperparameters for the BERT model.
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension of transformer representations.
    pub d_model: usize,
    /// Number of transformer encoder layers.
    pub num_layers: usize,
    /// Number of attention heads in multi-head self-attention.
    pub num_heads: usize,
    /// Inner hidden dimension of the Feed-Forward Network (FFN).
    pub d_ff: usize,
    /// Maximum sequence length for position embeddings.
    pub max_position_embeddings: usize,
    /// Number of segment / token-type categories (e.g. 2 for sentence A and sentence B).
    pub type_vocab_size: usize,
    /// Epsilon value for layer normalization.
    pub layer_norm_eps: f32,
}

impl BertConfig {
    /// Pre-configured compact BERT model for rapid training and experimentation.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 350,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            d_ff: 160,
            max_position_embeddings: 128,
            type_vocab_size: 2,
            layer_norm_eps: 1e-6,
        }
    }
}

/// BERT Embeddings module combining Word, Position, and Token-Type (Segment) embeddings.
#[derive(Clone)]
pub struct BertEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub config: BertConfig,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Self {
        Self {
            word_embeddings: Embedding::new(config.vocab_size, config.d_model),
            position_embeddings: Embedding::new(config.max_position_embeddings, config.d_model),
            token_type_embeddings: Embedding::new(config.type_vocab_size, config.d_model),
            layer_norm: LayerNorm::with_eps(config.d_model, config.layer_norm_eps),
            config: config.clone(),
        }
    }

    /// Computes combined embeddings for input tokens [B, T], optional token-types [B, T], and positions.
    pub fn forward_embeddings(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "BertEmbeddings forward (expected 2D input_ids [B, T])",
                shapes: vec![shape],
            });
        }

        let (b, t) = (shape[0], shape[1]);
        if t > self.config.max_position_embeddings {
            return Err(EngineError::IncompatibleShapes {
                op: "BertEmbeddings (sequence length exceeds max_position_embeddings)",
                shapes: vec![shape, vec![self.config.max_position_embeddings]],
            });
        }

        // 1. Word Token Embeddings -> [B, T, d_model]
        let words = self.word_embeddings.forward(input_ids)?;

        // 2. Position Embeddings -> [B, T, d_model]
        let mut pos_indices = Vec::with_capacity(b * t);
        for _ in 0..b {
            for pos in 0..t {
                pos_indices.push(pos as f32);
            }
        }
        let pos_tensor = Tensor::new(RawTensor::from_vec(pos_indices, vec![b, t]), false);
        let positions = self.position_embeddings.forward(&pos_tensor)?;

        // 3. Token-Type / Segment Embeddings -> [B, T, d_model]
        let types = if let Some(type_ids) = token_type_ids {
            self.token_type_embeddings.forward(type_ids)?
        } else {
            let zeros = Tensor::new(RawTensor::zeros(&[b, t]), false);
            self.token_type_embeddings.forward(&zeros)?
        };

        // 4. Sum all embeddings and apply LayerNorm
        let embeds = words.add(&positions)?.add(&types)?;
        self.layer_norm.forward(&embeds)
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_embeddings(input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.word_embeddings.parameters());
        params.extend(self.position_embeddings.parameters());
        params.extend(self.token_type_embeddings.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

/// A single BERT Transformer Encoder Layer (Bidirectional Self-Attention + GELU FFN).
#[derive(Clone)]
pub struct BertLayer {
    pub attention: MultiHeadAttention,
    pub attention_norm: LayerNorm,
    pub intermediate: Linear,
    pub output_dense: Linear,
    pub output_norm: LayerNorm,
}

impl BertLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, eps: f32) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads, false), // Bidirectional
            attention_norm: LayerNorm::with_eps(d_model, eps),
            intermediate: Linear::new(d_model, d_ff),
            output_dense: Linear::new(d_ff, d_model),
            output_norm: LayerNorm::with_eps(d_model, eps),
        }
    }

    pub fn forward_layer(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Bidirectional Self-Attention with Residual & LayerNorm
        let attn_out = self.attention.forward_attention(x)?;
        let x = self.attention_norm.forward(&x.add(&attn_out)?)?;

        // 2. GELU FFN with Residual & LayerNorm
        let inter = self.intermediate.forward(&x)?.gelu()?;
        let ffn_out = self.output_dense.forward(&inter)?;
        self.output_norm.forward(&x.add(&ffn_out)?)
    }
}

impl Module for BertLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_layer(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.attention_norm.parameters());
        params.extend(self.intermediate.parameters());
        params.extend(self.output_dense.parameters());
        params.extend(self.output_norm.parameters());
        params
    }
}

/// Stack of BERT Transformer Encoder Layers.
#[derive(Clone)]
pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(BertLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.layer_norm_eps,
            ));
        }
        Self { layers }
    }

    pub fn forward_encoder(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut x = hidden_states.clone();
        for layer in &self.layers {
            x = layer.forward_layer(&x)?;
        }
        Ok(x)
    }
}

impl Module for BertEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_encoder(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

/// BERT Pooler extracting dense sentence representations from the `[CLS]` token.
#[derive(Clone)]
pub struct BertPooler {
    pub dense: Linear,
}

impl BertPooler {
    pub fn new(d_model: usize) -> Self {
        Self {
            dense: Linear::new(d_model, d_model),
        }
    }

    /// Extracts the first token `[CLS]` from sequence representations [B, T, d_model] and applies tanh.
    pub fn forward_pooler(&self, sequence_output: &Tensor) -> Result<Tensor> {
        let shape = sequence_output.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "BertPooler forward (expected 3D sequence_output [B, T, d_model])",
                shapes: vec![shape],
            });
        }

        let (b, _t, d_model) = (shape[0], shape[1], shape[2]);
        // Slice [CLS] token at index 0 along dimension 1 -> [B, 1, d_model]
        let cls_token = sequence_output.slice(1, 0, 1)?;
        let cls_2d = cls_token.reshape(&[b, d_model])?;
        self.dense.forward(&cls_2d)?.tanh()
    }
}

impl Module for BertPooler {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_pooler(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.dense.parameters()
    }
}

/// The base BERT Model returning sequence token representations and pooled sentence representations.
#[derive(Clone)]
pub struct BertModel {
    pub embeddings: BertEmbeddings,
    pub encoder: BertEncoder,
    pub pooler: BertPooler,
    pub config: BertConfig,
}

impl BertModel {
    pub fn new(config: BertConfig) -> Self {
        let embeddings = BertEmbeddings::new(&config);
        let encoder = BertEncoder::new(&config);
        let pooler = BertPooler::new(config.d_model);
        Self {
            embeddings,
            encoder,
            pooler,
            config,
        }
    }

    /// Forward pass returning `(sequence_output [B, T, d_model], pooled_output [B, d_model])`.
    pub fn forward_bert(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let embeds = self
            .embeddings
            .forward_embeddings(input_ids, token_type_ids)?;
        let seq_out = self.encoder.forward_encoder(&embeds)?;
        let pooled_out = self.pooler.forward_pooler(&seq_out)?;
        Ok((seq_out, pooled_out))
    }
}

impl Module for BertModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (seq_out, _) = self.forward_bert(input, None)?;
        Ok(seq_out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.embeddings.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.pooler.parameters());
        params
    }
}

/// BERT Model with a span classification head for Extractive Question Answering (SQuAD style).
#[derive(Clone)]
pub struct BertForQuestionAnswering {
    pub bert: BertModel,
    pub qa_outputs: Linear,
}

impl BertForQuestionAnswering {
    pub fn new(config: BertConfig) -> Self {
        let d_model = config.d_model;
        let bert = BertModel::new(config);
        // Linear projecting d_model -> 2 (start_logit, end_logit)
        let qa_outputs = Linear::new(d_model, 2);
        Self { bert, qa_outputs }
    }

    /// Computes start and end logits for question answering span extraction.
    /// Returns `(start_logits [B, T], end_logits [B, T])`.
    pub fn forward_qa(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (seq_out, _) = self.bert.forward_bert(input_ids, token_type_ids)?;
        let shape = seq_out.shape();
        let (b, t, _d) = (shape[0], shape[1], shape[2]);

        // Logits: [B, T, 2]
        let logits = self.qa_outputs.forward(&seq_out)?;

        // Slice start logits (index 0) and end logits (index 1) along dimension 2
        let start_slice = logits.slice(2, 0, 1)?;
        let end_slice = logits.slice(2, 1, 2)?;

        let start_logits = start_slice.reshape(&[b, t])?;
        let end_logits = end_slice.reshape(&[b, t])?;

        Ok((start_logits, end_logits))
    }
}

impl Module for BertForQuestionAnswering {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (start, _end) = self.forward_qa(input, None)?;
        Ok(start)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.bert.parameters());
        params.extend(self.qa_outputs.parameters());
        params
    }
}

/// BERT Model specialized for computing normalized semantic text embeddings.
#[derive(Clone)]
pub struct BertForSequenceEmbedding {
    pub bert: BertModel,
}

impl BertForSequenceEmbedding {
    pub fn new(config: BertConfig) -> Self {
        Self {
            bert: BertModel::new(config),
        }
    }

    /// Computes pooled sentence embeddings of shape [B, d_model].
    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, pooled_out) = self.bert.forward_bert(input_ids, token_type_ids)?;
        Ok(pooled_out)
    }

    /// Computes cosine similarity between two sentence embedding vectors.
    pub fn cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
        assert_eq!(u.len(), v.len(), "Vectors must have identical length");
        let mut dot = 0.0f32;
        let mut norm_u = 0.0f32;
        let mut norm_v = 0.0f32;

        for (&a, &b) in u.iter().zip(v.iter()) {
            dot += a * b;
            norm_u += a * a;
            norm_v += b * b;
        }

        let denom = (norm_u.sqrt() * norm_v.sqrt()).max(1e-9);
        dot / denom
    }
}

impl Module for BertForSequenceEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_embedding(input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.bert.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_model_forward_and_backward() {
        let config = BertConfig {
            vocab_size: 100,
            d_model: 32,
            num_layers: 2,
            num_heads: 2,
            d_ff: 64,
            max_position_embeddings: 64,
            type_vocab_size: 2,
            layer_norm_eps: 1e-6,
        };

        let bert = BertModel::new(config);

        // Input token IDs: [Batch=2, SeqLen=8]
        let input_raw = RawTensor::from_slice(
            &[
                2.0, 15.0, 30.0, 3.0, 45.0, 60.0, 75.0, 3.0, // [CLS] Q [SEP] Ctx [SEP]
                2.0, 10.0, 20.0, 3.0, 40.0, 50.0, 60.0, 3.0,
            ],
            &[2, 8],
        );
        let input_ids = Tensor::new(input_raw, false);

        // Token types: 0 for Question, 1 for Context
        let types_raw = RawTensor::from_slice(
            &[
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            ],
            &[2, 8],
        );
        let token_type_ids = Tensor::new(types_raw, false);

        let (seq_out, pooled_out) = bert
            .forward_bert(&input_ids, Some(&token_type_ids))
            .unwrap();
        assert_eq!(seq_out.shape(), &[2, 8, 32]);
        assert_eq!(pooled_out.shape(), &[2, 32]);

        // Autograd backward check
        let loss = seq_out.sum_all();
        loss.backward();

        assert!(bert.embeddings.word_embeddings.weight.grad().is_some());
        assert!(bert.embeddings.position_embeddings.weight.grad().is_some());
        assert!(bert
            .embeddings
            .token_type_embeddings
            .weight
            .grad()
            .is_some());
    }

    #[test]
    fn test_bert_qa_head_forward_and_backward() {
        let config = BertConfig::tiny();
        let qa_model = BertForQuestionAnswering::new(config);

        let input_ids = Tensor::new(
            RawTensor::from_slice(&[2.0, 10.0, 3.0, 20.0, 30.0, 3.0], &[1, 6]),
            false,
        );
        let (start_logits, end_logits) = qa_model.forward_qa(&input_ids, None).unwrap();

        assert_eq!(start_logits.shape(), &[1, 6]);
        assert_eq!(end_logits.shape(), &[1, 6]);

        let loss = start_logits.sum_all().add(&end_logits.sum_all()).unwrap();
        loss.backward();

        assert!(qa_model.qa_outputs.weight.grad().is_some());
    }
}
