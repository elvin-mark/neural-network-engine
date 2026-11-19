//! Qwen3.5 Hybrid Architecture (`Qwen/Qwen3.5-0.8B` and compatible models).
//!
//! Combines:
//! - Gated DeltaNet (`linear_attention`) layers with causal depthwise Conv1d, L2-normalized Q/K,
//!   recurrent gated delta rule, and gated RMSNorm.
//! - Grouped-Query Full Attention (`full_attention`) layers with per-head Q/K RMSNorm,
//!   output gating (`sigmoid(gate)`), and partial Rotary Position Embeddings (RoPE).
//! - Zero-centered RMSNorm (`1.0 + weight`) and SwiGLU MLP blocks.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::tensor::RawTensor;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Layer mixer type in Qwen3.5 hybrid decoder blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3_5LayerType {
    LinearAttention,
    FullAttention,
}

/// Configuration hyperparameters for Qwen3.5 text models.
#[derive(Clone, Debug)]
pub struct Qwen3_5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub layer_types: Vec<Qwen3_5LayerType>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_id: usize,
}

impl Qwen3_5Config {
    /// Official `Qwen/Qwen3.5-0.8B` text backbone configuration.
    pub fn qwen3_5_0_8b() -> Self {
        let num_hidden_layers = 24;
        let layer_types = (0..num_hidden_layers)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    Qwen3_5LayerType::FullAttention
                } else {
                    Qwen3_5LayerType::LinearAttention
                }
            })
            .collect();

        Self {
            vocab_size: 248320,
            hidden_size: 1024,
            intermediate_size: 3584,
            num_hidden_layers,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            layer_types,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            eos_token_id: 248044,
        }
    }

    /// Small Qwen3.5 configuration for fast unit testing and local experimentation.
    pub fn mini(vocab_size: usize, max_position_embeddings: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            max_position_embeddings,
            tie_word_embeddings: true,
            eos_token_id: 0,
        }
    }

    /// Number of rotary dimensions per head in full-attention layers.
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32 * self.partial_rotary_factor) as usize) & !1
    }
}

/// Zero-centered RMSNorm used by Qwen3.5:
/// `y = x / sqrt(mean(x^2) + eps) * (1.0 + weight)`.
#[derive(Clone)]
pub struct Qwen3_5RMSNorm {
    pub normalized_dim: usize,
    pub weight: Tensor,
    pub eps: f32,
}

impl Qwen3_5RMSNorm {
    pub fn new(normalized_dim: usize, eps: f32) -> Self {
        Self {
            normalized_dim,
            weight: Tensor::zeros(&[normalized_dim], true),
            eps,
        }
    }
}

impl Module for Qwen3_5RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(EngineError::InvalidArgument(
                "Qwen3_5RMSNorm cannot be applied to a 0-D scalar tensor".to_string(),
            ));
        }
        let last_dim = *shape.last().unwrap();
        if last_dim != self.normalized_dim {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.normalized_dim],
                actual: vec![last_dim],
            });
        }

        if !crate::autograd::is_grad_enabled()
            || (!input.requires_grad() && !self.weight.requires_grad())
        {
            let in_raw = input.data().to_contiguous();
            let in_slice = in_raw.as_slice();
            let w_raw = self.weight.data().to_contiguous();
            let w_slice = w_raw.as_slice();
            let d = self.normalized_dim;
            let mut out_data = vec![0.0f32; in_slice.len()];

            out_data
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(row_idx, row_out)| {
                    let row_in = &in_slice[row_idx * d..(row_idx + 1) * d];
                    let sum_sq: f32 = row_in.iter().map(|&v| v * v).sum();
                    let inv_rms = 1.0 / ((sum_sq / d as f32) + self.eps).sqrt();
                    for i in 0..d {
                        row_out[i] = row_in[i] * inv_rms * (1.0 + w_slice[i]);
                    }
                });

            return Ok(Tensor::new(RawTensor::from_vec(out_data, shape), false));
        }

        let last_axis = input.ndim() - 1;
        let sq = input.powf(2.0)?;
        let mean_sq = sq.mean(last_axis, true)?;
        let inv_rms = mean_sq.add(&Tensor::scalar(self.eps, false))?.powf(-0.5)?;
        let norm = input.mul(&inv_rms)?;
        let one_plus_w = self.weight.add_scalar(1.0)?;
        norm.mul(&one_plus_w)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// Gated RMSNorm used at the output of `Qwen3_5GatedDeltaNet`:
/// `y = (x / sqrt(mean(x^2) + eps) * weight) * silu(gate)`.
#[derive(Clone)]
pub struct Qwen3_5RMSNormGated {
    pub normalized_dim: usize,
    pub weight: Tensor,
    pub eps: f32,
}

impl Qwen3_5RMSNormGated {
    pub fn new(normalized_dim: usize, eps: f32) -> Self {
        Self {
            normalized_dim,
            weight: Tensor::ones(&[normalized_dim], true),
            eps,
        }
    }

    pub fn forward(&self, input: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let last_dim = *shape.last().ok_or_else(|| {
            EngineError::InvalidArgument(
                "Qwen3_5RMSNormGated cannot be applied to a 0-D scalar tensor".to_string(),
            )
        })?;
        if last_dim != self.normalized_dim {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.normalized_dim],
                actual: vec![last_dim],
            });
        }

        let in_raw = input.data().to_contiguous();
        let in_slice = in_raw.as_slice();
        let gate_raw = gate.data().to_contiguous();
        let gate_slice = gate_raw.as_slice();
        if in_slice.len() != gate_slice.len() {
            return Err(EngineError::ShapeMismatch {
                expected: shape,
                actual: gate.shape(),
            });
        }
        let w_raw = self.weight.data().to_contiguous();
        let w_slice = w_raw.as_slice();
        let d = self.normalized_dim;
        let mut out_data = vec![0.0f32; in_slice.len()];

        out_data
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(row_idx, row_out)| {
                let offset = row_idx * d;
                let row_in = &in_slice[offset..offset + d];
                let row_gate = &gate_slice[offset..offset + d];
                let sum_sq: f32 = row_in.iter().map(|&v| v * v).sum();
                let inv_rms = 1.0 / ((sum_sq / d as f32) + self.eps).sqrt();
                for i in 0..d {
                    let g = row_gate[i];
                    let silu_g = g / (1.0 + (-g).exp());
                    row_out[i] = row_in[i] * inv_rms * w_slice[i] * silu_g;
                }
            });

        Ok(Tensor::new(RawTensor::from_vec(out_data, shape), false))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// Per-layer state for Qwen3.5 hybrid inference caching.
#[derive(Clone, Debug)]
pub enum Qwen3_5LayerCache {
    Linear {
        /// `[batch_size, conv_dim, conv_kernel_size]`
        conv_state: Option<RawTensor>,
        /// `[batch_size, num_v_heads, head_k_dim, head_v_dim]`
        recurrent_state: Option<RawTensor>,
    },
    Full {
        /// Cached `(K, V)` tensors of shape `[batch_size, num_kv_heads, past_seq_len, head_dim]`
        kv: Option<(Tensor, Tensor)>,
    },
}

/// Hybrid cache holding recurrent + conv states for linear-attention layers
/// and KV tensors for full-attention layers.
#[derive(Clone, Debug)]
pub struct Qwen3_5Cache {
    pub layers: Vec<Qwen3_5LayerCache>,
}

impl Qwen3_5Cache {
    pub fn new(config: &Qwen3_5Config) -> Self {
        let layers = config
            .layer_types
            .iter()
            .map(|lt| match lt {
                Qwen3_5LayerType::LinearAttention => Qwen3_5LayerCache::Linear {
                    conv_state: None,
                    recurrent_state: None,
                },
                Qwen3_5LayerType::FullAttention => Qwen3_5LayerCache::Full { kv: None },
            })
            .collect();
        Self { layers }
    }
}

/// Gated DeltaNet linear-attention layer (`linear_attention`).
#[derive(Clone)]
pub struct Qwen3_5GatedDeltaNet {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel_size: usize,
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Linear,
    pub in_proj_a: Linear,
    pub conv1d_weight: Tensor, // [conv_dim, conv_kernel_size]
    pub dt_bias: Tensor,       // [num_v_heads]
    pub a_log: Tensor,         // [num_v_heads]
    pub norm: Qwen3_5RMSNormGated,
    pub out_proj: Linear,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(config: &Qwen3_5Config) -> Self {
        let key_dim = config.linear_key_head_dim * config.linear_num_key_heads;
        let value_dim = config.linear_value_head_dim * config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;

        Self {
            hidden_size: config.hidden_size,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
            in_proj_qkv: Linear::without_bias(config.hidden_size, conv_dim),
            in_proj_z: Linear::without_bias(config.hidden_size, value_dim),
            in_proj_b: Linear::without_bias(config.hidden_size, config.linear_num_value_heads),
            in_proj_a: Linear::without_bias(config.hidden_size, config.linear_num_value_heads),
            conv1d_weight: Tensor::randn(
                &[conv_dim, config.linear_conv_kernel_dim],
                0.0,
                0.02,
                true,
            ),
            dt_bias: Tensor::ones(&[config.linear_num_value_heads], true),
            a_log: Tensor::zeros(&[config.linear_num_value_heads], true),
            norm: Qwen3_5RMSNormGated::new(config.linear_value_head_dim, config.rms_norm_eps),
            out_proj: Linear::without_bias(value_dim, config.hidden_size),
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut Qwen3_5LayerCache>,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(EngineError::InvalidArgument(format!(
                "Qwen3_5GatedDeltaNet expects 3D [B, T, D], got {:?}",
                shape
            )));
        }
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?; // [B, T, conv_dim]
        let z = self.in_proj_z.forward(hidden_states)?; // [B, T, value_dim]
        let b_proj = self.in_proj_b.forward(hidden_states)?; // [B, T, num_v_heads]
        let a_proj = self.in_proj_a.forward(hidden_states)?; // [B, T, num_v_heads]

        let qkv_raw = mixed_qkv.data().to_contiguous();
        let qkv_slice = qkv_raw.as_slice();
        let conv_w_raw = self.conv1d_weight.data().to_contiguous();
        let conv_w_slice = conv_w_raw.as_slice();

        let k_size = self.conv_kernel_size;
        let c_dim = self.conv_dim;

        // Extract mutable references to cached conv_state and recurrent_state if provided
        let (conv_state_opt, rec_state_opt) = match cache {
            Some(Qwen3_5LayerCache::Linear {
                conv_state,
                recurrent_state,
            }) => (Some(conv_state), Some(recurrent_state)),
            Some(_) => {
                return Err(EngineError::InvalidArgument(
                    "Expected Linear cache slot for Qwen3_5GatedDeltaNet".to_string(),
                ))
            }
            None => (None, None),
        };

        // 1. Causal depthwise Conv1d + SiLU
        let mut conv_out = vec![0.0f32; batch_size * seq_len * c_dim];
        match conv_state_opt {
            Some(conv_state_slot) => {
                if let Some(prev_conv) = conv_state_slot.as_mut() {
                    if seq_len == 1 {
                        // Fast single-token decode: shift left by 1 and append current token
                        let mut state_data = prev_conv.to_contiguous().as_slice().to_vec();
                        for b in 0..batch_size {
                            let st_b =
                                &mut state_data[b * c_dim * k_size..(b + 1) * c_dim * k_size];
                            let in_b = &qkv_slice[b * c_dim..(b + 1) * c_dim];
                            let out_b = &mut conv_out[b * c_dim..(b + 1) * c_dim];
                            for c in 0..c_dim {
                                let st_c = &mut st_b[c * k_size..(c + 1) * k_size];
                                for k in 0..(k_size - 1) {
                                    st_c[k] = st_c[k + 1];
                                }
                                st_c[k_size - 1] = in_b[c];

                                let w_c = &conv_w_slice[c * k_size..(c + 1) * k_size];
                                let mut acc = 0.0f32;
                                for k in 0..k_size {
                                    acc += st_c[k] * w_c[k];
                                }
                                out_b[c] = acc / (1.0 + (-acc).exp());
                            }
                        }
                        *prev_conv =
                            RawTensor::from_vec(state_data, vec![batch_size, c_dim, k_size]);
                    } else {
                        // Multi-token continuation with prior conv_state
                        let prev_contig = prev_conv.to_contiguous();
                        let prev_slice = prev_contig.as_slice();
                        let mut next_state = vec![0.0f32; batch_size * c_dim * k_size];
                        for b in 0..batch_size {
                            let prev_b = &prev_slice[b * c_dim * k_size..(b + 1) * c_dim * k_size];
                            for t in 0..seq_len {
                                for c in 0..c_dim {
                                    let w_c = &conv_w_slice[c * k_size..(c + 1) * k_size];
                                    let mut acc = 0.0f32;
                                    for (k, &w_val) in w_c.iter().enumerate().take(k_size) {
                                        let offset_t =
                                            t as isize + k as isize - (k_size as isize - 1);
                                        let val = if offset_t >= 0 {
                                            qkv_slice[(b * seq_len + offset_t as usize) * c_dim + c]
                                        } else {
                                            let prev_idx = k_size as isize + offset_t;
                                            if prev_idx >= 0 {
                                                prev_b[c * k_size + prev_idx as usize]
                                            } else {
                                                0.0
                                            }
                                        };
                                        acc += val * w_val;
                                    }
                                    conv_out[(b * seq_len + t) * c_dim + c] =
                                        acc / (1.0 + (-acc).exp());
                                }
                            }
                            for c in 0..c_dim {
                                for k in 0..k_size {
                                    let src_t = seq_len as isize - k_size as isize + k as isize;
                                    let val = if src_t >= 0 {
                                        qkv_slice[(b * seq_len + src_t as usize) * c_dim + c]
                                    } else {
                                        let prev_idx = k_size as isize + src_t;
                                        if prev_idx >= 0 {
                                            prev_b[c * k_size + prev_idx as usize]
                                        } else {
                                            0.0
                                        }
                                    };
                                    next_state[(b * c_dim + c) * k_size + k] = val;
                                }
                            }
                        }
                        *prev_conv =
                            RawTensor::from_vec(next_state, vec![batch_size, c_dim, k_size]);
                    }
                } else {
                    // Prefill: zero-padded causal conv1d + save last k_size inputs to conv_state
                    causal_conv1d_silu_prefill(
                        qkv_slice,
                        conv_w_slice,
                        batch_size,
                        seq_len,
                        c_dim,
                        k_size,
                        &mut conv_out,
                    );
                    let mut init_state = vec![0.0f32; batch_size * c_dim * k_size];
                    for b in 0..batch_size {
                        for c in 0..c_dim {
                            for k in 0..k_size {
                                let src_t = seq_len as isize - k_size as isize + k as isize;
                                if src_t >= 0 {
                                    init_state[(b * c_dim + c) * k_size + k] =
                                        qkv_slice[(b * seq_len + src_t as usize) * c_dim + c];
                                }
                            }
                        }
                    }
                    *conv_state_slot = Some(RawTensor::from_vec(
                        init_state,
                        vec![batch_size, c_dim, k_size],
                    ));
                }
            }
            None => {
                causal_conv1d_silu_prefill(
                    qkv_slice,
                    conv_w_slice,
                    batch_size,
                    seq_len,
                    c_dim,
                    k_size,
                    &mut conv_out,
                );
            }
        }

        // 2. Recurrent Gated Delta Rule
        let b_raw = b_proj.data().to_contiguous();
        let b_slice = b_raw.as_slice();
        let a_raw = a_proj.data().to_contiguous();
        let a_slice = a_raw.as_slice();
        let dt_bias_raw = self.dt_bias.data().to_contiguous();
        let dt_bias_slice = dt_bias_raw.as_slice();
        let a_log_raw = self.a_log.data().to_contiguous();
        let a_log_slice = a_log_raw.as_slice();

        let state_elems = batch_size * self.num_v_heads * self.head_k_dim * self.head_v_dim;
        let mut state_vec = match rec_state_opt.as_ref().and_then(|s| s.as_ref()) {
            Some(prev_rec) => prev_rec.to_contiguous().as_slice().to_vec(),
            None => vec![0.0f32; state_elems],
        };

        let mut core_out = vec![0.0f32; batch_size * seq_len * self.value_dim];
        let q_scale = 1.0 / (self.head_k_dim as f32).sqrt();
        let head_rep = self.num_v_heads / self.num_k_heads;
        let hk = self.head_k_dim;
        let hv = self.head_v_dim;
        let nh_v = self.num_v_heads;

        let mut q_norm = vec![0.0f32; hk];
        let mut k_norm = vec![0.0f32; hk];
        let mut kv_mem = vec![0.0f32; hv];
        let mut delta = vec![0.0f32; hv];

        for b in 0..batch_size {
            for h in 0..nh_v {
                let kh = h / head_rep;
                let a_exp = a_log_slice[h].exp();
                let dt_b = dt_bias_slice[h];
                let s_offset = (b * nh_v + h) * hk * hv;
                let s_mat = &mut state_vec[s_offset..s_offset + hk * hv];

                for t in 0..seq_len {
                    let qkv_base = (b * seq_len + t) * c_dim;
                    let q_raw = &conv_out[qkv_base + kh * hk..qkv_base + (kh + 1) * hk];
                    let k_raw = &conv_out[qkv_base + self.key_dim + kh * hk
                        ..qkv_base + self.key_dim + (kh + 1) * hk];
                    let v_raw = &conv_out[qkv_base + 2 * self.key_dim + h * hv
                        ..qkv_base + 2 * self.key_dim + (h + 1) * hv];

                    // L2-normalize q and k with eps = 1e-6, and scale q by 1 / sqrt(hk)
                    let q_sq: f32 = q_raw.iter().map(|&x| x * x).sum();
                    let q_inv = (1.0 / (q_sq + 1e-6).sqrt()) * q_scale;
                    for i in 0..hk {
                        q_norm[i] = q_raw[i] * q_inv;
                    }

                    let k_sq: f32 = k_raw.iter().map(|&x| x * x).sum();
                    let k_inv = 1.0 / (k_sq + 1e-6).sqrt();
                    for i in 0..hk {
                        k_norm[i] = k_raw[i] * k_inv;
                    }

                    let b_val = b_slice[(b * seq_len + t) * nh_v + h];
                    let beta_t = 1.0 / (1.0 + (-b_val).exp());

                    let a_val = a_slice[(b * seq_len + t) * nh_v + h] + dt_b;
                    let softplus_a = if a_val > 20.0 {
                        a_val
                    } else {
                        (1.0 + a_val.exp()).ln()
                    };
                    let decay_t = (-a_exp * softplus_a).exp();

                    // 1. Decay state: S = S * decay_t
                    for val in s_mat.iter_mut() {
                        *val *= decay_t;
                    }

                    // 2. kv_mem = sum_i S[i, j] * k_t[i]
                    kv_mem.fill(0.0);
                    for i in 0..hk {
                        let k_i = k_norm[i];
                        let row = &s_mat[i * hv..(i + 1) * hv];
                        for j in 0..hv {
                            kv_mem[j] += row[j] * k_i;
                        }
                    }

                    // 3. delta = (v_t - kv_mem) * beta_t
                    for j in 0..hv {
                        delta[j] = (v_raw[j] - kv_mem[j]) * beta_t;
                    }

                    // 4. Update state and compute output:
                    //    S[i, j] += k_t[i] * delta[j]
                    //    out[j]  += S[i, j] * q_t[i]
                    let out_base = (b * seq_len + t) * self.value_dim + h * hv;
                    let out_head = &mut core_out[out_base..out_base + hv];
                    out_head.fill(0.0);
                    for i in 0..hk {
                        let k_i = k_norm[i];
                        let q_i = q_norm[i];
                        let row = &mut s_mat[i * hv..(i + 1) * hv];
                        for j in 0..hv {
                            let updated = row[j] + k_i * delta[j];
                            row[j] = updated;
                            out_head[j] += updated * q_i;
                        }
                    }
                }
            }
        }

        if let Some(rec_slot) = rec_state_opt {
            *rec_slot = Some(RawTensor::from_vec(
                state_vec,
                vec![batch_size, nh_v, hk, hv],
            ));
        }

        // 3. Gated RMSNorm per head + output projection
        let core_tensor = Tensor::new(
            RawTensor::from_vec(core_out, vec![batch_size * seq_len * nh_v, hv]),
            false,
        );
        let z_reshaped = z.reshape(&[batch_size * seq_len * nh_v, hv])?;
        let normed = self.norm.forward(&core_tensor, &z_reshaped)?;
        let normed_3d = normed.reshape(&[batch_size, seq_len, self.value_dim])?;

        self.out_proj.forward(&normed_3d)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.in_proj_qkv.parameters());
        params.extend(self.in_proj_z.parameters());
        params.extend(self.in_proj_b.parameters());
        params.extend(self.in_proj_a.parameters());
        params.push(self.conv1d_weight.clone());
        params.push(self.dt_bias.clone());
        params.push(self.a_log.clone());
        params.extend(self.norm.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

fn causal_conv1d_silu_prefill(
    input: &[f32],
    weight: &[f32],
    batch_size: usize,
    seq_len: usize,
    c_dim: usize,
    k_size: usize,
    output: &mut [f32],
) {
    for b in 0..batch_size {
        for t in 0..seq_len {
            for c in 0..c_dim {
                let w_c = &weight[c * k_size..(c + 1) * k_size];
                let mut acc = 0.0f32;
                for (k, &w_val) in w_c.iter().enumerate().take(k_size) {
                    let src_t = t as isize + k as isize - (k_size as isize - 1);
                    if src_t >= 0 {
                        acc += input[(b * seq_len + src_t as usize) * c_dim + c] * w_val;
                    }
                }
                output[(b * seq_len + t) * c_dim + c] = acc / (1.0 + (-acc).exp());
            }
        }
    }
}

/// Applies partial Rotary Position Embeddings (RoPE) on-the-fly to a 4D tensor `[B, H, T, D]`.
/// Only the first `rotary_dim` dimensions of `D` are rotated; the remaining `D - rotary_dim`
/// dimensions are passed through unchanged.
fn apply_partial_rope(
    x: &Tensor,
    start_pos: usize,
    rotary_dim: usize,
    rope_theta: f32,
) -> Result<Tensor> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(EngineError::InvalidArgument(format!(
            "apply_partial_rope expects 4D [B, H, T, D], got {:?}",
            shape
        )));
    }
    let (b, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
    if rotary_dim == 0 {
        return Ok(x.clone());
    }
    if rotary_dim > d || rotary_dim % 2 != 0 {
        return Err(EngineError::InvalidArgument(format!(
            "Invalid rotary_dim {} for head_dim {}",
            rotary_dim, d
        )));
    }

    let half_rot = rotary_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_rot);
    for i in 0..half_rot {
        let exponent = (2 * i) as f32 / rotary_dim as f32;
        inv_freq.push(1.0 / rope_theta.powf(exponent));
    }

    let x_raw = x.data().to_contiguous();
    let x_slice = x_raw.as_slice();
    let mut out_data = x_slice.to_vec();

    for batch in 0..b {
        for head in 0..h {
            for pos_offset in 0..t {
                let pos = (start_pos + pos_offset) as f32;
                let base = ((batch * h + head) * t + pos_offset) * d;
                for i in 0..half_rot {
                    let freq = pos * inv_freq[i];
                    let c = freq.cos();
                    let s = freq.sin();
                    let x1 = x_slice[base + i];
                    let x2 = x_slice[base + i + half_rot];
                    out_data[base + i] = x1 * c - x2 * s;
                    out_data[base + i + half_rot] = x2 * c + x1 * s;
                }
            }
        }
    }

    Ok(Tensor::new(RawTensor::from_vec(out_data, shape), false))
}

/// Full Grouped-Query Attention (`full_attention`) with Q/K RMSNorm, partial RoPE, and output gate.
#[derive(Clone)]
pub struct Qwen3_5Attention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Qwen3_5RMSNorm,
    pub k_norm: Qwen3_5RMSNorm,
}

impl Qwen3_5Attention {
    pub fn new(config: &Qwen3_5Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim(),
            rope_theta: config.rope_theta,
            q_proj: Linear::without_bias(
                config.hidden_size,
                config.num_attention_heads * config.head_dim * 2,
            ),
            k_proj: Linear::without_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            ),
            v_proj: Linear::without_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            ),
            o_proj: Linear::without_bias(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            ),
            q_norm: Qwen3_5RMSNorm::new(config.head_dim, config.rms_norm_eps),
            k_norm: Qwen3_5RMSNorm::new(config.head_dim, config.rms_norm_eps),
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3_5LayerCache>,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(EngineError::InvalidArgument(format!(
                "Qwen3_5Attention expects 3D [B, T, D], got {:?}",
                shape
            )));
        }
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let q_and_gate = self.q_proj.forward(hidden_states)?.reshape(&[
            b,
            t,
            self.num_heads,
            self.head_dim * 2,
        ])?;
        let q_raw = q_and_gate.slice(3, 0, self.head_dim)?;
        let gate = q_and_gate
            .slice(3, self.head_dim, self.head_dim * 2)?
            .reshape(&[b, t, self.num_heads * self.head_dim])?;

        let k_raw = self.k_proj.forward(hidden_states)?.reshape(&[
            b,
            t,
            self.num_kv_heads,
            self.head_dim,
        ])?;
        let v_raw = self.v_proj.forward(hidden_states)?.reshape(&[
            b,
            t,
            self.num_kv_heads,
            self.head_dim,
        ])?;

        let q = self.q_norm.forward(&q_raw)?.transpose(1, 2)?; // [B, H, T, D]
        let k = self.k_norm.forward(&k_raw)?.transpose(1, 2)?; // [B, H_kv, T, D]
        let v = v_raw.transpose(1, 2)?; // [B, H_kv, T, D]

        let q = apply_partial_rope(&q, start_pos, self.rotary_dim, self.rope_theta)?;
        let k = apply_partial_rope(&k, start_pos, self.rotary_dim, self.rope_theta)?;

        // Update KV cache if provided
        let (k_full, v_full) = match cache {
            Some(Qwen3_5LayerCache::Full { kv }) => {
                if let Some((prev_k, prev_v)) = kv.as_ref() {
                    if prev_k.shape()[0] > 0 {
                        let cat_k = Tensor::cat(&[prev_k, &k], 2)?;
                        let cat_v = Tensor::cat(&[prev_v, &v], 2)?;
                        *kv = Some((cat_k.clone(), cat_v.clone()));
                        (cat_k, cat_v)
                    } else {
                        *kv = Some((k.clone(), v.clone()));
                        (k, v)
                    }
                } else {
                    *kv = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
            Some(_) => {
                return Err(EngineError::InvalidArgument(
                    "Expected Full cache slot for Qwen3_5Attention".to_string(),
                ))
            }
            None => (k, v),
        };

        let total_t = k_full.shape()[2];
        let num_groups = self.num_heads / self.num_kv_heads;
        let (k_attn, v_attn) = if num_groups > 1 {
            let k_exp =
                k_full
                    .unsqueeze(2)?
                    .reshape(&[b, self.num_kv_heads, 1, total_t, self.head_dim])?;
            let mut k_slices = Vec::with_capacity(num_groups);
            let v_exp =
                v_full
                    .unsqueeze(2)?
                    .reshape(&[b, self.num_kv_heads, 1, total_t, self.head_dim])?;
            let mut v_slices = Vec::with_capacity(num_groups);
            for _ in 0..num_groups {
                k_slices.push(k_exp.clone());
                v_slices.push(v_exp.clone());
            }
            let k_refs: Vec<&Tensor> = k_slices.iter().collect();
            let v_refs: Vec<&Tensor> = v_slices.iter().collect();
            let k_rep =
                Tensor::cat(&k_refs, 2)?.reshape(&[b, self.num_heads, total_t, self.head_dim])?;
            let v_rep =
                Tensor::cat(&v_refs, 2)?.reshape(&[b, self.num_heads, total_t, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_full, v_full)
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let k_t = k_attn.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?.mul_scalar(scale)?;

        if t > 1 {
            let mut mask_data = vec![0.0f32; t * total_t];
            for i in 0..t {
                let query_pos = start_pos + i;
                for j in 0..total_t {
                    if j > query_pos {
                        mask_data[i * total_t + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::new(
                RawTensor::from_vec(mask_data, vec![1, 1, t, total_t]),
                false,
            );
            scores = scores.add(&mask)?;
        }

        let probs = scores.softmax(3)?;
        let attn_out = probs.matmul(&v_attn)?.transpose(1, 2)?.reshape(&[
            b,
            t,
            self.num_heads * self.head_dim,
        ])?;

        let gated = attn_out.mul(&gate.sigmoid()?)?;
        self.o_proj.forward(&gated)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params.extend(self.q_norm.parameters());
        params.extend(self.k_norm.parameters());
        params
    }
}

/// SwiGLU Feed-Forward Network used in Qwen3.5 decoder layers.
#[derive(Clone)]
pub struct Qwen3_5MLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Qwen3_5MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::without_bias(hidden_size, intermediate_size),
            up_proj: Linear::without_bias(hidden_size, intermediate_size),
            down_proj: Linear::without_bias(intermediate_size, hidden_size),
        }
    }
}

impl Module for Qwen3_5MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(input)?.silu()?;
        let up = self.up_proj.forward(input)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

/// Token mixer variant inside a `Qwen3_5DecoderLayer`.
#[derive(Clone)]
pub enum Qwen3_5TokenMixer {
    Linear(Qwen3_5GatedDeltaNet),
    Full(Qwen3_5Attention),
}

/// Single Qwen3.5 Transformer/DeltaNet hybrid decoder layer.
#[derive(Clone)]
pub struct Qwen3_5DecoderLayer {
    pub input_layernorm: Qwen3_5RMSNorm,
    pub mixer: Qwen3_5TokenMixer,
    pub post_attention_layernorm: Qwen3_5RMSNorm,
    pub mlp: Qwen3_5MLP,
}

impl Qwen3_5DecoderLayer {
    pub fn new(config: &Qwen3_5Config, layer_type: Qwen3_5LayerType) -> Self {
        let mixer = match layer_type {
            Qwen3_5LayerType::LinearAttention => {
                Qwen3_5TokenMixer::Linear(Qwen3_5GatedDeltaNet::new(config))
            }
            Qwen3_5LayerType::FullAttention => {
                Qwen3_5TokenMixer::Full(Qwen3_5Attention::new(config))
            }
        };

        Self {
            input_layernorm: Qwen3_5RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            mixer,
            post_attention_layernorm: Qwen3_5RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp: Qwen3_5MLP::new(config.hidden_size, config.intermediate_size),
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3_5LayerCache>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            Qwen3_5TokenMixer::Linear(delta_net) => delta_net.forward_cached(&normed, cache)?,
            Qwen3_5TokenMixer::Full(attn) => attn.forward_cached(&normed, start_pos, cache)?,
        };
        let h = hidden_states.add(&mixed)?;
        let mlp_in = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        h.add(&mlp_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.input_layernorm.parameters();
        match &self.mixer {
            Qwen3_5TokenMixer::Linear(delta_net) => params.extend(delta_net.parameters()),
            Qwen3_5TokenMixer::Full(attn) => params.extend(attn.parameters()),
        }
        params.extend(self.post_attention_layernorm.parameters());
        params.extend(self.mlp.parameters());
        params
    }
}

/// Complete Qwen3.5 Causal Language Model.
#[derive(Clone)]
pub struct Qwen3_5Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    pub norm: Qwen3_5RMSNorm,
    pub lm_head: Linear,
    pub config: Qwen3_5Config,
}

impl Qwen3_5Model {
    pub fn new(config: Qwen3_5Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for &lt in &config.layer_types {
            layers.push(Qwen3_5DecoderLayer::new(&config, lt));
        }

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            norm: Qwen3_5RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            lm_head: Linear::without_bias(config.hidden_size, config.vocab_size),
            config,
        }
    }

    /// Forward pass for token indices `[batch_size * seq_len]` with optional hybrid cache.
    pub fn forward_tokens_cached(
        &self,
        token_indices: &[usize],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
        mut cache: Option<&mut Qwen3_5Cache>,
    ) -> Result<Tensor> {
        if token_indices.len() != batch_size * seq_len {
            return Err(EngineError::InvalidArgument(format!(
                "token_indices length {} must match batch_size ({}) * seq_len ({})",
                token_indices.len(),
                batch_size,
                seq_len
            )));
        }
        if start_pos + seq_len > self.config.max_position_embeddings {
            return Err(EngineError::InvalidArgument(format!(
                "Sequence position {} exceeds max_position_embeddings {}",
                start_pos + seq_len,
                self.config.max_position_embeddings
            )));
        }

        let tok = self.embed_tokens.forward_indices(token_indices)?;
        let mut x = tok.reshape(&[batch_size, seq_len, self.config.hidden_size])?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = match cache.as_deref_mut() {
                Some(c) => c.layers.get_mut(layer_idx),
                None => None,
            };
            x = layer.forward_cached(&x, start_pos, layer_cache)?;
        }

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Forward pass without caching.
    pub fn forward_tokens(
        &self,
        token_indices: &[usize],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_tokens_cached(token_indices, batch_size, seq_len, start_pos, None)
    }

    /// Generates tokens autoregressively using the hybrid recurrent/KV cache.
    pub fn generate_cached(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidArgument(
                "prompt_tokens cannot be empty".to_string(),
            ));
        }
        let mut tokens = prompt_tokens.to_vec();
        if max_new_tokens == 0 {
            return Ok(tokens);
        }

        let mut cache = Qwen3_5Cache::new(&self.config);

        let logits =
            self.forward_tokens_cached(prompt_tokens, 1, prompt_tokens.len(), 0, Some(&mut cache))?;
        let last_logits = logits
            .slice(1, prompt_tokens.len() - 1, prompt_tokens.len())?
            .squeeze(1)?
            .squeeze(0)?;
        let mut next_token = sample_greedy_or_temp(&last_logits, temperature)?;
        tokens.push(next_token);

        for _ in 1..max_new_tokens {
            if tokens.len() >= self.config.max_position_embeddings {
                break;
            }
            let start_pos = tokens.len() - 1;
            let logits =
                self.forward_tokens_cached(&[next_token], 1, 1, start_pos, Some(&mut cache))?;
            let step_logits = logits.squeeze(1)?.squeeze(0)?;
            next_token = sample_greedy_or_temp(&step_logits, temperature)?;
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Autoregressive text generation with streaming callback and top-k/top-p sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream<R: rand::Rng, F: FnMut(usize, &str) -> Result<bool>>(
        &self,
        tokenizer: &crate::tokenizer::HfTokenizer,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        eos_token_id: Option<usize>,
        rng: &mut R,
        mut on_token: F,
    ) -> Result<Vec<usize>> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidArgument(
                "prompt_tokens cannot be empty".to_string(),
            ));
        }
        let mut tokens = prompt_tokens.to_vec();
        if max_new_tokens == 0 {
            return Ok(tokens);
        }

        let mut cache = Qwen3_5Cache::new(&self.config);

        let logits =
            self.forward_tokens_cached(prompt_tokens, 1, prompt_tokens.len(), 0, Some(&mut cache))?;
        let last_logits = logits
            .slice(1, prompt_tokens.len() - 1, prompt_tokens.len())?
            .squeeze(1)?
            .squeeze(0)?;

        let mut next_token = crate::utils::sample_token(
            last_logits.data().to_contiguous().as_slice(),
            temperature,
            top_k,
            top_p,
            rng,
        );
        tokens.push(next_token);

        let piece = tokenizer.decode_token(next_token);
        let keep_going = on_token(next_token, &piece)?;
        if !keep_going || eos_token_id == Some(next_token) {
            return Ok(tokens);
        }

        for _ in 1..max_new_tokens {
            if tokens.len() >= self.config.max_position_embeddings {
                break;
            }
            let start_pos = tokens.len() - 1;
            let logits =
                self.forward_tokens_cached(&[next_token], 1, 1, start_pos, Some(&mut cache))?;
            let step_logits = logits.squeeze(1)?.squeeze(0)?;
            next_token = crate::utils::sample_token(
                step_logits.data().to_contiguous().as_slice(),
                temperature,
                top_k,
                top_p,
                rng,
            );
            tokens.push(next_token);

            let piece = tokenizer.decode_token(next_token);
            let keep_going = on_token(next_token, &piece)?;
            if !keep_going || eos_token_id == Some(next_token) {
                break;
            }
        }

        Ok(tokens)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.embed_tokens.parameters();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    /// Loads model weights from an in-memory tensor map.
    /// Supports both converted checkpoint names (`embed_tokens.weight`, `layers.0...`)
    /// and raw HuggingFace `Qwen3_5ForConditionalGeneration` names (`model.language_model.embed_tokens.weight`, ...).
    pub fn load_weights(&mut self, weights: &HashMap<String, RawTensor>) -> Result<()> {
        let prefix = if weights.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if weights.contains_key("model.embed_tokens.weight") {
            "model."
        } else {
            ""
        };

        let get_tensor = |key: &str| -> Result<&RawTensor> {
            let full_key = format!("{}{}", prefix, key);
            weights.get(&full_key).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "Missing expected weight '{}' in weights map",
                    full_key
                ))
            })
        };

        let embed_raw = get_tensor("embed_tokens.weight")?.clone();
        self.embed_tokens.weight.set_data(embed_raw.clone());

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .input_layernorm
                .weight
                .set_data(get_tensor(&format!("layers.{}.input_layernorm.weight", i))?.clone());
            layer.post_attention_layernorm.weight.set_data(
                get_tensor(&format!("layers.{}.post_attention_layernorm.weight", i))?.clone(),
            );
            layer
                .mlp
                .gate_proj
                .weight
                .set_data(get_tensor(&format!("layers.{}.mlp.gate_proj.weight", i))?.clone());
            layer
                .mlp
                .up_proj
                .weight
                .set_data(get_tensor(&format!("layers.{}.mlp.up_proj.weight", i))?.clone());
            layer
                .mlp
                .down_proj
                .weight
                .set_data(get_tensor(&format!("layers.{}.mlp.down_proj.weight", i))?.clone());

            match &mut layer.mixer {
                Qwen3_5TokenMixer::Linear(dn) => {
                    dn.in_proj_qkv.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.in_proj_qkv.weight", i))?
                            .clone(),
                    );
                    dn.in_proj_z.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.in_proj_z.weight", i))?.clone(),
                    );
                    dn.in_proj_b.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.in_proj_b.weight", i))?.clone(),
                    );
                    dn.in_proj_a.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.in_proj_a.weight", i))?.clone(),
                    );
                    dn.dt_bias.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.dt_bias", i))?.clone(),
                    );
                    dn.a_log
                        .set_data(get_tensor(&format!("layers.{}.linear_attn.A_log", i))?.clone());
                    dn.norm.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.norm.weight", i))?.clone(),
                    );
                    dn.out_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.linear_attn.out_proj.weight", i))?.clone(),
                    );

                    // conv1d.weight may be [conv_dim, 1, K] in HF or [conv_dim, K]
                    let conv_raw = get_tensor(&format!("layers.{}.linear_attn.conv1d.weight", i))?;
                    let conv_squeezed = if conv_raw.shape().len() == 3 && conv_raw.shape()[1] == 1 {
                        RawTensor::from_vec(
                            conv_raw.to_contiguous().as_slice().to_vec(),
                            vec![conv_raw.shape()[0], conv_raw.shape()[2]],
                        )
                    } else {
                        conv_raw.clone()
                    };
                    dn.conv1d_weight.set_data(conv_squeezed);
                }
                Qwen3_5TokenMixer::Full(attn) => {
                    attn.q_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.q_proj.weight", i))?.clone(),
                    );
                    attn.k_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.k_proj.weight", i))?.clone(),
                    );
                    attn.v_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.v_proj.weight", i))?.clone(),
                    );
                    attn.o_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.o_proj.weight", i))?.clone(),
                    );
                    attn.q_norm.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.q_norm.weight", i))?.clone(),
                    );
                    attn.k_norm.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.k_norm.weight", i))?.clone(),
                    );
                }
            }
        }

        self.norm
            .weight
            .set_data(get_tensor("norm.weight")?.clone());

        if let Some(lm_head) = weights.get("lm_head.weight") {
            self.lm_head.weight.set_data(lm_head.clone());
        } else if self.config.tie_word_embeddings {
            self.lm_head.weight.set_data(embed_raw);
        } else {
            return Err(EngineError::InvalidArgument(
                "Missing expected weight 'lm_head.weight' and tie_word_embeddings is false"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Loads model weights directly from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let weights = crate::io::load_safetensors(path)?;
        self.load_weights(&weights)
    }
}

fn sample_greedy_or_temp(logits: &Tensor, temperature: f32) -> Result<usize> {
    let contig = logits.data().to_contiguous();
    let slice = contig.as_slice();
    if temperature <= 1e-5 {
        let best = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        return Ok(best);
    }

    let mut rng = rand::thread_rng();
    Ok(crate::utils::sample_token(
        slice,
        temperature,
        None,
        None,
        &mut rng,
    ))
}
