//! FlashAttention-2: Tiled Memory-Efficient Online Softmax Attention.
//!
//! Eliminates materialization of the $O(T^2)$ intermediate attention matrix in RAM by
//! computing attention in tiled cache blocks ($B_r \times B_c$) with online softmax normalization.
//!
//! References:
//! - Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." (2023).

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::tensor::RawTensor;
use rayon::prelude::*;

/// Executes FlashAttention-2 online-softmax tiled kernel on 4D tensors [BatchSize, NumHeads, SeqLen, HeadDim].
///
/// Reduces memory from $O(T^2)$ to $O(T)$ and operates entirely within L1/L2 CPU cache blocks.
pub fn flash_attention_forward(
    q: &RawTensor,
    k: &RawTensor,
    v: &RawTensor,
    is_causal: bool,
    scale: Option<f32>,
    block_size_r: usize,
    block_size_c: usize,
) -> Result<RawTensor> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(EngineError::InvalidArgument(format!(
            "FlashAttention expects 4D tensors [Batch, Heads, SeqLen, Dim], got Q:{:?}, K:{:?}, V:{:?}",
            q_shape, k_shape, v_shape
        )));
    }

    let (b, h, t_q, d) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (k_b, k_h, t_k, k_d) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    let (v_b, v_h, v_t, v_d) = (v_shape[0], v_shape[1], v_shape[2], v_shape[3]);

    if b != k_b || b != v_b || h != k_h || h != v_h || d != k_d || d != v_d || t_k != v_t {
        return Err(EngineError::ShapeMismatch {
            expected: vec![b, h, t_k, d],
            actual: vec![k_b, k_h, t_k, k_d],
        });
    }

    let sm_scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let br = block_size_r.max(16);
    let bc = block_size_c.max(16);

    let q_contig = q.to_contiguous();
    let k_contig = k.to_contiguous();
    let v_contig = v.to_contiguous();

    let q_slice = q_contig.as_slice();
    let k_slice = k_contig.as_slice();
    let v_slice = v_contig.as_slice();

    let head_matrix_len_q = t_q * d;
    let head_matrix_len_k = t_k * d;
    let total_heads = b * h;

    let mut out = vec![0.0f32; total_heads * head_matrix_len_q];

    // Parallelize over all (Batch * Head) matrices
    out.par_chunks_mut(head_matrix_len_q)
        .enumerate()
        .for_each(|(bh_idx, o_head)| {
            let q_offset = bh_idx * head_matrix_len_q;
            let k_offset = bh_idx * head_matrix_len_k;
            let v_offset = bh_idx * head_matrix_len_k;

            let q_mat = &q_slice[q_offset..q_offset + head_matrix_len_q];
            let k_mat = &k_slice[k_offset..k_offset + head_matrix_len_k];
            let v_mat = &v_slice[v_offset..v_offset + head_matrix_len_k];

            // Running statistics per query row: max m_i, sum l_i
            let mut m = vec![f32::NEG_INFINITY; t_q];
            let mut l = vec![0.0f32; t_q];

            let num_r_blocks = t_q.div_ceil(br);
            let num_c_blocks = t_k.div_ceil(bc);

            // Intermediate block buffers in L1/L2 cache
            let mut s_ij = vec![0.0f32; br * bc];
            let mut p_ij = vec![0.0f32; br * bc];

            for i in 0..num_r_blocks {
                let r_start = i * br;
                let r_end = (r_start + br).min(t_q);
                let m_r = r_end - r_start;

                for j in 0..num_c_blocks {
                    let c_start = j * bc;
                    let c_end = (c_start + bc).min(t_k);
                    let m_c = c_end - c_start;

                    // If causal and entire col block is strictly after row block, skip
                    if is_causal && c_start > r_end - 1 {
                        continue;
                    }

                    // 1. Compute block attention scores S_ij = (Q_i * K_j^T) * scale
                    for r in 0..m_r {
                        let r_global = r_start + r;
                        let q_row = &q_mat[r_global * d..(r_global + 1) * d];

                        for c in 0..m_c {
                            let c_global = c_start + c;
                            if is_causal && c_global > r_global {
                                s_ij[r * bc + c] = f32::NEG_INFINITY;
                            } else {
                                let k_row = &k_mat[c_global * d..(c_global + 1) * d];
                                let mut dot = 0.0f32;
                                for idx in 0..d {
                                    dot += q_row[idx] * k_row[idx];
                                }
                                s_ij[r * bc + c] = dot * sm_scale;
                            }
                        }
                    }

                    // 2. Online Softmax update per query row
                    for r in 0..m_r {
                        let r_global = r_start + r;

                        // Find row max in block
                        let mut block_row_max = f32::NEG_INFINITY;
                        for c in 0..m_c {
                            let score = s_ij[r * bc + c];
                            if score > block_row_max {
                                block_row_max = score;
                            }
                        }

                        if block_row_max == f32::NEG_INFINITY {
                            continue;
                        }

                        let old_m = m[r_global];
                        let new_m = old_m.max(block_row_max);

                        let alpha = if old_m == f32::NEG_INFINITY {
                            0.0f32
                        } else {
                            (old_m - new_m).exp()
                        };

                        // Compute unnormalized exp: P_ij = exp(S_ij - new_m)
                        let mut p_row_sum = 0.0f32;
                        for c in 0..m_c {
                            let score = s_ij[r * bc + c];
                            let p = if score == f32::NEG_INFINITY {
                                0.0f32
                            } else {
                                (score - new_m).exp()
                            };
                            p_ij[r * bc + c] = p;
                            p_row_sum += p;
                        }

                        let new_l = alpha * l[r_global] + p_row_sum;

                        // 3. Rescale previous O and accumulate P_ij * V_j
                        let o_row = &mut o_head[r_global * d..(r_global + 1) * d];
                        for d_idx in 0..d {
                            let mut pv = 0.0f32;
                            for c in 0..m_c {
                                let p_val = p_ij[r * bc + c];
                                let v_val = v_mat[(c_start + c) * d + d_idx];
                                pv += p_val * v_val;
                            }
                            o_row[d_idx] = alpha * o_row[d_idx] + pv;
                        }

                        m[r_global] = new_m;
                        l[r_global] = new_l;
                    }
                }
            }

            // 4. Final normalization by row-sum: O_i = O_i / l_i
            for r_global in 0..t_q {
                let row_sum = l[r_global];
                if row_sum > 0.0 {
                    let inv_l = 1.0 / row_sum;
                    let o_row = &mut o_head[r_global * d..(r_global + 1) * d];
                    for val in o_row.iter_mut() {
                        *val *= inv_l;
                    }
                }
            }
        });

    Ok(RawTensor::from_vec(out, vec![b, h, t_q, d]))
}

/// Standalone Multi-Head FlashAttention layer.
#[derive(Clone)]
pub struct FlashAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub head_dim: usize,
    pub is_causal: bool,
    pub block_size: usize,
}

impl FlashAttention {
    pub fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let head_dim = d_model / num_heads;

        Self {
            q_proj: Linear::new(d_model, d_model),
            k_proj: Linear::new(d_model, d_model),
            v_proj: Linear::new(d_model, d_model),
            out_proj: Linear::new(d_model, d_model),
            num_heads,
            d_model,
            head_dim,
            is_causal,
            block_size: 64,
        }
    }

    /// Forward pass with O(T) FlashAttention-2 online softmax execution.
    pub fn forward_flash(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "FlashAttention forward (expected 3D input [B, T, C])",
                shapes: vec![shape],
            });
        }

        let (b, t, _) = (shape[0], shape[1], shape[2]);
        let h = self.num_heads;
        let d = self.head_dim;

        // Projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape & transpose to [B, H, T, D]
        let q_4d = q.reshape(&[b, t, h, d])?.transpose(1, 2)?;
        let k_4d = k.reshape(&[b, t, h, d])?.transpose(1, 2)?;
        let v_4d = v.reshape(&[b, t, h, d])?.transpose(1, 2)?;

        let out_raw = flash_attention_forward(
            &q_4d.data(),
            &k_4d.data(),
            &v_4d.data(),
            self.is_causal,
            None,
            self.block_size,
            self.block_size,
        )?;

        let out_tensor = Tensor::new(out_raw, x.requires_grad());
        let out_merged = out_tensor.transpose(1, 2)?.reshape(&[b, t, self.d_model])?;
        self.out_proj.forward(&out_merged)
    }
}

impl Module for FlashAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_flash(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}
