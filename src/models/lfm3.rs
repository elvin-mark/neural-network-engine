//! Liquid AI LFM3 Hybrid Architecture (`LiquidAI/LFM3-1.7B` and compatible models).
//!
//! Combines:
//! - `Lfm3ShortConv` (`conv`): gated depthwise causal 1D convolution (width 7) with `B, C, x` gating.
//! - `Lfm3Attention` (`full_attention` and `sliding_attention`): Grouped-Query Attention with
//!   Q/K RMSNorm, per-head sigmoid output gating (`gate_proj`), optional sliding window, and
//!   LFM3 zero-tail partial RoPE.
//! - Sandwich-normalized (`operator_norm` + `post_operator_norm`, `ffn_norm` + `post_ffn_norm`)
//!   decoder blocks with SwiGLU (`w1`, `w2`, `w3`) MLPs.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::RMSNorm;
use crate::tensor::RawTensor;
use std::collections::HashMap;
use std::path::Path;

/// Layer mixer type in LFM3 hybrid decoder blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lfm3LayerType {
    Conv,
    FullAttention,
    SlidingAttention,
}

/// Configuration hyperparameters for LFM3 models.
#[derive(Clone, Debug)]
pub struct Lfm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<Lfm3LayerType>,
    pub norm_eps: f32,
    pub full_rope_theta: f32,
    pub full_partial_rotary_factor: f32,
    pub sliding_rope_theta: f32,
    pub sliding_partial_rotary_factor: f32,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_sandwich_norm: bool,
    pub eos_token_id: usize,
}

impl Lfm3Config {
    /// Official `LiquidAI/LFM3-1.7B` configuration.
    pub fn lfm3_1_7b() -> Self {
        let layer_types = vec![
            Lfm3LayerType::Conv,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::SlidingAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::SlidingAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::SlidingAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::SlidingAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::SlidingAttention,
            Lfm3LayerType::Conv,
            Lfm3LayerType::FullAttention,
            Lfm3LayerType::Conv,
        ];

        Self {
            vocab_size: 128_000,
            hidden_size: 2048,
            intermediate_size: 7680,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            conv_l_cache: 7,
            conv_bias: false,
            layer_types,
            norm_eps: 1e-5,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.5,
            sliding_rope_theta: 10_000.0,
            sliding_partial_rotary_factor: 1.0,
            sliding_window: 256,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            use_sandwich_norm: true,
            eos_token_id: 124_900,
        }
    }

    /// Small LFM3 configuration for fast unit testing and parity verification.
    pub fn mini(vocab_size: usize, max_position_embeddings: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            conv_l_cache: 7,
            conv_bias: false,
            layer_types: vec![
                Lfm3LayerType::Conv,
                Lfm3LayerType::FullAttention,
                Lfm3LayerType::SlidingAttention,
                Lfm3LayerType::Conv,
            ],
            norm_eps: 1e-5,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.5,
            sliding_rope_theta: 10_000.0,
            sliding_partial_rotary_factor: 1.0,
            sliding_window: 4,
            max_position_embeddings,
            tie_word_embeddings: true,
            use_sandwich_norm: true,
            eos_token_id: 0,
        }
    }
}

/// Per-layer inference cache for LFM3 (`Conv` state or `Attention` KV tensors).
#[derive(Clone, Debug)]
pub enum Lfm3LayerCache {
    Conv {
        /// `[batch_size, hidden_size, conv_l_cache]`
        conv_state: Option<RawTensor>,
    },
    Attention {
        /// Cached `(K, V)` tensors of shape `[batch_size, num_kv_heads, past_seq_len, head_dim]`
        kv: Option<(Tensor, Tensor)>,
    },
}

/// Hybrid cache holding conv states and attention KV tensors across all LFM3 layers.
#[derive(Clone, Debug)]
pub struct Lfm3Cache {
    pub layers: Vec<Lfm3LayerCache>,
}

impl Lfm3Cache {
    pub fn new(config: &Lfm3Config) -> Self {
        let layers = config
            .layer_types
            .iter()
            .map(|lt| match lt {
                Lfm3LayerType::Conv => Lfm3LayerCache::Conv { conv_state: None },
                Lfm3LayerType::FullAttention | Lfm3LayerType::SlidingAttention => {
                    Lfm3LayerCache::Attention { kv: None }
                }
            })
            .collect();
        Self { layers }
    }
}

/// LFM3 gated short depthwise convolution layer (`conv`).
#[derive(Clone)]
pub struct Lfm3ShortConv {
    pub hidden_size: usize,
    pub conv_kernel_size: usize,
    pub in_proj: Linear,
    pub conv_weight: Tensor, // [hidden_size, conv_kernel_size]
    pub out_proj: Linear,
}

impl Lfm3ShortConv {
    pub fn new(config: &Lfm3Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            conv_kernel_size: config.conv_l_cache,
            in_proj: Linear::without_bias(config.hidden_size, 3 * config.hidden_size),
            conv_weight: Tensor::randn(&[config.hidden_size, config.conv_l_cache], 0.0, 0.02, true),
            out_proj: Linear::without_bias(config.hidden_size, config.hidden_size),
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut Lfm3LayerCache>,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(EngineError::InvalidArgument(format!(
                "Lfm3ShortConv expects 3D [B, T, D], got {:?}",
                shape
            )));
        }
        let (batch_size, seq_len, d) = (shape[0], shape[1], shape[2]);
        let k_size = self.conv_kernel_size;

        let bcx = self.in_proj.forward(hidden_states)?; // [B, T, 3 * D]
        let bcx_raw = bcx.data().to_contiguous();
        let bcx_slice = bcx_raw.as_slice();

        // Compute B * x for each [b, t, c]
        let mut bx = vec![0.0f32; batch_size * seq_len * d];
        for b in 0..batch_size {
            for t in 0..seq_len {
                let src_base = (b * seq_len + t) * 3 * d;
                let dst_base = (b * seq_len + t) * d;
                for c in 0..d {
                    bx[dst_base + c] = bcx_slice[src_base + c] * bcx_slice[src_base + 2 * d + c];
                }
            }
        }

        let conv_w_raw = self.conv_weight.data().to_contiguous();
        let conv_w_slice = conv_w_raw.as_slice();
        let mut conv_out = vec![0.0f32; batch_size * seq_len * d];

        let conv_state_opt = match cache {
            Some(Lfm3LayerCache::Conv { conv_state }) => Some(conv_state),
            Some(_) => {
                return Err(EngineError::InvalidArgument(
                    "Expected Conv cache slot for Lfm3ShortConv".to_string(),
                ))
            }
            None => None,
        };

        match conv_state_opt {
            Some(conv_state_slot) => {
                if let Some(prev_conv) = conv_state_slot.as_mut() {
                    if seq_len == 1 {
                        let mut state_data = prev_conv.to_contiguous().as_slice().to_vec();
                        for b in 0..batch_size {
                            let st_b = &mut state_data[b * d * k_size..(b + 1) * d * k_size];
                            let in_b = &bx[b * d..(b + 1) * d];
                            let out_b = &mut conv_out[b * d..(b + 1) * d];
                            for c in 0..d {
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
                                out_b[c] = acc;
                            }
                        }
                        *prev_conv = RawTensor::from_vec(state_data, vec![batch_size, d, k_size]);
                    } else {
                        let prev_contig = prev_conv.to_contiguous();
                        let prev_slice = prev_contig.as_slice();
                        let mut next_state = vec![0.0f32; batch_size * d * k_size];
                        for b in 0..batch_size {
                            let prev_b = &prev_slice[b * d * k_size..(b + 1) * d * k_size];
                            for t in 0..seq_len {
                                for c in 0..d {
                                    let w_c = &conv_w_slice[c * k_size..(c + 1) * k_size];
                                    let mut acc = 0.0f32;
                                    for (k, &w_val) in w_c.iter().enumerate().take(k_size) {
                                        let offset_t =
                                            t as isize + k as isize - (k_size as isize - 1);
                                        let val = if offset_t >= 0 {
                                            bx[(b * seq_len + offset_t as usize) * d + c]
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
                                    conv_out[(b * seq_len + t) * d + c] = acc;
                                }
                            }
                            for c in 0..d {
                                for k in 0..k_size {
                                    let src_t = seq_len as isize - k_size as isize + k as isize;
                                    let val = if src_t >= 0 {
                                        bx[(b * seq_len + src_t as usize) * d + c]
                                    } else {
                                        let prev_idx = k_size as isize + src_t;
                                        if prev_idx >= 0 {
                                            prev_b[c * k_size + prev_idx as usize]
                                        } else {
                                            0.0
                                        }
                                    };
                                    next_state[(b * d + c) * k_size + k] = val;
                                }
                            }
                        }
                        *prev_conv = RawTensor::from_vec(next_state, vec![batch_size, d, k_size]);
                    }
                } else {
                    causal_conv1d_prefill(
                        &bx,
                        conv_w_slice,
                        batch_size,
                        seq_len,
                        d,
                        k_size,
                        &mut conv_out,
                    );
                    let mut init_state = vec![0.0f32; batch_size * d * k_size];
                    for b in 0..batch_size {
                        for c in 0..d {
                            for k in 0..k_size {
                                let src_t = seq_len as isize - k_size as isize + k as isize;
                                if src_t >= 0 {
                                    init_state[(b * d + c) * k_size + k] =
                                        bx[(b * seq_len + src_t as usize) * d + c];
                                }
                            }
                        }
                    }
                    *conv_state_slot =
                        Some(RawTensor::from_vec(init_state, vec![batch_size, d, k_size]));
                }
            }
            None => {
                causal_conv1d_prefill(
                    &bx,
                    conv_w_slice,
                    batch_size,
                    seq_len,
                    d,
                    k_size,
                    &mut conv_out,
                );
            }
        }

        // Apply C gate: y = C * conv_out
        for b in 0..batch_size {
            for t in 0..seq_len {
                let src_base = (b * seq_len + t) * 3 * d + d;
                let dst_base = (b * seq_len + t) * d;
                for c in 0..d {
                    conv_out[dst_base + c] *= bcx_slice[src_base + c];
                }
            }
        }

        let y = Tensor::new(
            RawTensor::from_vec(conv_out, vec![batch_size, seq_len, d]),
            false,
        );
        self.out_proj.forward(&y)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.in_proj.parameters();
        params.push(self.conv_weight.clone());
        params.extend(self.out_proj.parameters());
        params
    }
}

fn causal_conv1d_prefill(
    input: &[f32],
    weight: &[f32],
    batch_size: usize,
    seq_len: usize,
    d: usize,
    k_size: usize,
    output: &mut [f32],
) {
    for b in 0..batch_size {
        for t in 0..seq_len {
            for c in 0..d {
                let w_c = &weight[c * k_size..(c + 1) * k_size];
                let mut acc = 0.0f32;
                for (k, &w_val) in w_c.iter().enumerate().take(k_size) {
                    let src_t = t as isize + k as isize - (k_size as isize - 1);
                    if src_t >= 0 {
                        acc += input[(b * seq_len + src_t as usize) * d + c] * w_val;
                    }
                }
                output[(b * seq_len + t) * d + c] = acc;
            }
        }
    }
}

/// Applies LFM3 zero-tail partial RoPE on-the-fly to `[B, H, T, D]`.
/// Only the first `rope_angles = floor(partial_rotary_factor * D) / 2` pairs `(i, i + D/2)`
/// are rotated using the full-head-dim frequency progression `theta^(2i / D)`;
/// the remaining pairs have zero frequency (`cos = 1, sin = 0`) and pass through unchanged.
fn apply_lfm3_rope(
    x: &Tensor,
    start_pos: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
) -> Result<Tensor> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(EngineError::InvalidArgument(format!(
            "apply_lfm3_rope expects 4D [B, H, T, D], got {:?}",
            shape
        )));
    }
    let (b, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
    let half_d = d / 2;
    let rope_angles = ((partial_rotary_factor * d as f32) as usize) / 2;
    if rope_angles == 0 {
        return Ok(x.clone());
    }

    let mut inv_freq = Vec::with_capacity(rope_angles);
    for i in 0..rope_angles {
        let exponent = (2 * i) as f32 / d as f32;
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
                for i in 0..rope_angles {
                    let freq = pos * inv_freq[i];
                    let c = freq.cos();
                    let s = freq.sin();
                    let x1 = x_slice[base + i];
                    let x2 = x_slice[base + i + half_d];
                    out_data[base + i] = x1 * c - x2 * s;
                    out_data[base + i + half_d] = x2 * c + x1 * s;
                }
            }
        }
    }

    Ok(Tensor::new(RawTensor::from_vec(out_data, shape), false))
}

/// Grouped-Query Attention with Q/K RMSNorm, per-head sigmoid output gate,
/// LFM3 zero-tail partial RoPE, and optional sliding-window masking.
#[derive(Clone)]
pub struct Lfm3Attention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub sliding_window: Option<usize>,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub gate_proj: Linear,
}

impl Lfm3Attention {
    pub fn new(config: &Lfm3Config, layer_type: Lfm3LayerType) -> Self {
        let (rope_theta, partial_rotary_factor, sliding_window) = match layer_type {
            Lfm3LayerType::FullAttention => (
                config.full_rope_theta,
                config.full_partial_rotary_factor,
                None,
            ),
            Lfm3LayerType::SlidingAttention => (
                config.sliding_rope_theta,
                config.sliding_partial_rotary_factor,
                Some(config.sliding_window),
            ),
            Lfm3LayerType::Conv => unreachable!("Lfm3Attention created for Conv layer"),
        };

        Self {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta,
            partial_rotary_factor,
            sliding_window,
            q_proj: Linear::without_bias(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
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
            q_norm: RMSNorm::with_eps(config.head_dim, config.norm_eps),
            k_norm: RMSNorm::with_eps(config.head_dim, config.norm_eps),
            gate_proj: Linear::without_bias(config.hidden_size, config.num_attention_heads),
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        cache: Option<&mut Lfm3LayerCache>,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(EngineError::InvalidArgument(format!(
                "Lfm3Attention expects 3D [B, T, D], got {:?}",
                shape
            )));
        }
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let q_raw =
            self.q_proj
                .forward(hidden_states)?
                .reshape(&[b, t, self.num_heads, self.head_dim])?;
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

        let q = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k_raw)?.transpose(1, 2)?;
        let v = v_raw.transpose(1, 2)?;

        let q = apply_lfm3_rope(&q, start_pos, self.partial_rotary_factor, self.rope_theta)?;
        let k = apply_lfm3_rope(&k, start_pos, self.partial_rotary_factor, self.rope_theta)?;

        let (k_full, v_full) = match cache {
            Some(Lfm3LayerCache::Attention { kv }) => {
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
                    "Expected Attention cache slot for Lfm3Attention".to_string(),
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
            let v_exp =
                v_full
                    .unsqueeze(2)?
                    .reshape(&[b, self.num_kv_heads, 1, total_t, self.head_dim])?;
            let mut k_slices = Vec::with_capacity(num_groups);
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

        // Causal mask + optional sliding window mask
        let needs_mask = t > 1 || self.sliding_window.is_some_and(|w| total_t > w);
        if needs_mask {
            let mut mask_data = vec![0.0f32; t * total_t];
            for i in 0..t {
                let query_pos = start_pos + i;
                for j in 0..total_t {
                    let future = j > query_pos;
                    let outside_window = self
                        .sliding_window
                        .is_some_and(|w| query_pos >= w && j <= query_pos - w);
                    if future || outside_window {
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
        let attn_out = probs.matmul(&v_attn)?.transpose(1, 2)?; // [B, T, H, D]

        let gate = self
            .gate_proj
            .forward(hidden_states)?
            .sigmoid()?
            .unsqueeze(3)?; // [B, T, H, 1]
        let gated = attn_out
            .mul(&gate)?
            .reshape(&[b, t, self.num_heads * self.head_dim])?;

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
        params.extend(self.gate_proj.parameters());
        params
    }
}

/// SwiGLU Feed-Forward Network (`w1`, `w2`, `w3`) used in LFM3.
#[derive(Clone)]
pub struct Lfm3MLP {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl Lfm3MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            w1: Linear::without_bias(hidden_size, intermediate_size),
            w2: Linear::without_bias(intermediate_size, hidden_size),
            w3: Linear::without_bias(hidden_size, intermediate_size),
        }
    }
}

impl Module for Lfm3MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(input)?.silu()?;
        let up = self.w3.forward(input)?;
        self.w2.forward(&gate.mul(&up)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.w1.parameters());
        params.extend(self.w2.parameters());
        params.extend(self.w3.parameters());
        params
    }
}

/// Token mixer variant inside an `Lfm3DecoderLayer`.
#[derive(Clone)]
pub enum Lfm3TokenMixer {
    Conv(Lfm3ShortConv),
    Attention(Lfm3Attention),
}

/// Single LFM3 hybrid decoder layer with optional sandwich normalization.
#[derive(Clone)]
pub struct Lfm3DecoderLayer {
    pub operator_norm: RMSNorm,
    pub mixer: Lfm3TokenMixer,
    pub post_operator_norm: Option<RMSNorm>,
    pub ffn_norm: RMSNorm,
    pub feed_forward: Lfm3MLP,
    pub post_ffn_norm: Option<RMSNorm>,
}

impl Lfm3DecoderLayer {
    pub fn new(config: &Lfm3Config, layer_type: Lfm3LayerType) -> Self {
        let mixer = match layer_type {
            Lfm3LayerType::Conv => Lfm3TokenMixer::Conv(Lfm3ShortConv::new(config)),
            Lfm3LayerType::FullAttention | Lfm3LayerType::SlidingAttention => {
                Lfm3TokenMixer::Attention(Lfm3Attention::new(config, layer_type))
            }
        };

        let (post_operator_norm, post_ffn_norm) = if config.use_sandwich_norm {
            (
                Some(RMSNorm::with_eps(config.hidden_size, config.norm_eps)),
                Some(RMSNorm::with_eps(config.hidden_size, config.norm_eps)),
            )
        } else {
            (None, None)
        };

        Self {
            operator_norm: RMSNorm::with_eps(config.hidden_size, config.norm_eps),
            mixer,
            post_operator_norm,
            ffn_norm: RMSNorm::with_eps(config.hidden_size, config.norm_eps),
            feed_forward: Lfm3MLP::new(config.hidden_size, config.intermediate_size),
            post_ffn_norm,
        }
    }

    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        cache: Option<&mut Lfm3LayerCache>,
    ) -> Result<Tensor> {
        let normed = self.operator_norm.forward(hidden_states)?;
        let mut mixed = match &self.mixer {
            Lfm3TokenMixer::Conv(conv) => conv.forward_cached(&normed, cache)?,
            Lfm3TokenMixer::Attention(attn) => attn.forward_cached(&normed, start_pos, cache)?,
        };
        if let Some(post_op) = &self.post_operator_norm {
            mixed = post_op.forward(&mixed)?;
        }
        let h = hidden_states.add(&mixed)?;

        let ffn_in = self.ffn_norm.forward(&h)?;
        let mut ffn_out = self.feed_forward.forward(&ffn_in)?;
        if let Some(post_ffn) = &self.post_ffn_norm {
            ffn_out = post_ffn.forward(&ffn_out)?;
        }
        h.add(&ffn_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.operator_norm.parameters();
        match &self.mixer {
            Lfm3TokenMixer::Conv(conv) => params.extend(conv.parameters()),
            Lfm3TokenMixer::Attention(attn) => params.extend(attn.parameters()),
        }
        if let Some(post_op) = &self.post_operator_norm {
            params.extend(post_op.parameters());
        }
        params.extend(self.ffn_norm.parameters());
        params.extend(self.feed_forward.parameters());
        if let Some(post_ffn) = &self.post_ffn_norm {
            params.extend(post_ffn.parameters());
        }
        params
    }
}

/// Complete LFM3 Causal Language Model.
#[derive(Clone)]
pub struct Lfm3Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<Lfm3DecoderLayer>,
    pub embedding_norm: RMSNorm,
    pub lm_head: Linear,
    pub config: Lfm3Config,
}

impl Lfm3Model {
    pub fn new(config: Lfm3Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for &lt in &config.layer_types {
            layers.push(Lfm3DecoderLayer::new(&config, lt));
        }

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            embedding_norm: RMSNorm::with_eps(config.hidden_size, config.norm_eps),
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
        mut cache: Option<&mut Lfm3Cache>,
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

        let x = self.embedding_norm.forward(&x)?;
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

    /// Generates tokens autoregressively using the hybrid conv/KV cache.
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

        let mut cache = Lfm3Cache::new(&self.config);

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

        let mut cache = Lfm3Cache::new(&self.config);

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
        params.extend(self.embedding_norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    /// Loads model weights from an in-memory tensor map.
    /// Supports both converted checkpoint names (`embed_tokens.weight`, `layers.0...`)
    /// and raw HuggingFace `Lfm3ForCausalLM` names (`model.embed_tokens.weight`, `model.layers.0...`).
    pub fn load_weights(&mut self, weights: &HashMap<String, RawTensor>) -> Result<()> {
        let prefix = if weights.contains_key("model.embed_tokens.weight") {
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
                .operator_norm
                .weight
                .set_data(get_tensor(&format!("layers.{}.operator_norm.weight", i))?.clone());
            layer
                .ffn_norm
                .weight
                .set_data(get_tensor(&format!("layers.{}.ffn_norm.weight", i))?.clone());
            if let Some(post_op) = &mut layer.post_operator_norm {
                post_op.weight.set_data(
                    get_tensor(&format!("layers.{}.post_operator_norm.weight", i))?.clone(),
                );
            }
            if let Some(post_ffn) = &mut layer.post_ffn_norm {
                post_ffn
                    .weight
                    .set_data(get_tensor(&format!("layers.{}.post_ffn_norm.weight", i))?.clone());
            }
            layer
                .feed_forward
                .w1
                .weight
                .set_data(get_tensor(&format!("layers.{}.feed_forward.w1.weight", i))?.clone());
            layer
                .feed_forward
                .w2
                .weight
                .set_data(get_tensor(&format!("layers.{}.feed_forward.w2.weight", i))?.clone());
            layer
                .feed_forward
                .w3
                .weight
                .set_data(get_tensor(&format!("layers.{}.feed_forward.w3.weight", i))?.clone());

            match &mut layer.mixer {
                Lfm3TokenMixer::Conv(conv) => {
                    conv.in_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.conv.in_proj.weight", i))?.clone(),
                    );
                    conv.out_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.conv.out_proj.weight", i))?.clone(),
                    );
                    let conv_raw = get_tensor(&format!("layers.{}.conv.conv.weight", i))?;
                    let conv_squeezed = if conv_raw.shape().len() == 3 && conv_raw.shape()[1] == 1 {
                        RawTensor::from_vec(
                            conv_raw.to_contiguous().as_slice().to_vec(),
                            vec![conv_raw.shape()[0], conv_raw.shape()[2]],
                        )
                    } else {
                        conv_raw.clone()
                    };
                    conv.conv_weight.set_data(conv_squeezed);
                }
                Lfm3TokenMixer::Attention(attn) => {
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
                    attn.gate_proj.weight.set_data(
                        get_tensor(&format!("layers.{}.self_attn.gate_proj.weight", i))?.clone(),
                    );
                }
            }
        }

        self.embedding_norm
            .weight
            .set_data(get_tensor("embedding_norm.weight")?.clone());

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
