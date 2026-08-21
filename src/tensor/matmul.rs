//! High-performance tiled, cache-blocked, multi-threaded matrix multiplication (GEMM).
//! Features 4x16 register-tiled microkernel, M=1 parallel GEMV fast-path, and direct A*B^T transposed multiplication.

use crate::error::{EngineError, Result};
use crate::tensor::shape::{broadcast_shapes, compute_c_contiguous_strides, numel};
use crate::tensor::RawTensor;
use rayon::prelude::*;

const BLOCK_M: usize = 64;
const BLOCK_K: usize = 256;
const BLOCK_N: usize = 256;

// Microkernel tile dimensions
const MR: usize = 4;
const NR: usize = 16;

/// Performs 2D matrix multiplication C = A * B on contiguous row-major slices.
/// Employs 4x16 register-tiling, cache-blocking, and Rayon parallelization.
pub fn gemm_2d_contiguous(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // --- Fast-path: M = 1 GEMV (Matrix-Vector Multiplication, common in LLM decoding) ---
    if m == 1 {
        gemv_m1_parallel(k, n, a, b, c);
        return;
    }

    // Parallelize across row chunks of A and C
    let num_threads = rayon::current_num_threads();
    let chunk_size = (m.div_ceil(num_threads)).div_ceil(MR) * MR;
    let chunk_size = chunk_size.max(BLOCK_M);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = (c_chunk.len() / n).min(m.saturating_sub(m_start));
            let a_chunk = &a[m_start * k..(m_start + m_len) * k];

            // Tiled GEMM with L1/L2 cache blocking
            for k_block in (0..k).step_by(BLOCK_K) {
                let k_end = (k_block + BLOCK_K).min(k);

                for n_block in (0..n).step_by(BLOCK_N) {
                    let n_end = (n_block + BLOCK_N).min(n);

                    // 4x16 Register-Blocked Microkernel Loop
                    let mut i = 0;
                    while i + MR <= m_len {
                        let c_row0 = i * n;
                        let c_row1 = (i + 1) * n;
                        let c_row2 = (i + 2) * n;
                        let c_row3 = (i + 3) * n;

                        let a_row0 = &a_chunk[i * k..];
                        let a_row1 = &a_chunk[(i + 1) * k..];
                        let a_row2 = &a_chunk[(i + 2) * k..];
                        let a_row3 = &a_chunk[(i + 3) * k..];

                        let mut j = n_block;
                        while j + NR <= n_end {
                            // 4x16 accumulator tile in registers / stack
                            let mut c0 = [0.0f32; NR];
                            let mut c1 = [0.0f32; NR];
                            let mut c2 = [0.0f32; NR];
                            let mut c3 = [0.0f32; NR];

                            for kk in k_block..k_end {
                                let a0 = a_row0[kk];
                                let a1 = a_row1[kk];
                                let a2 = a_row2[kk];
                                let a3 = a_row3[kk];

                                let b_row = &b[kk * n + j..];

                                // 16 unrolled FMA operations per step k
                                for d in 0..NR {
                                    let b_val = b_row[d];
                                    c0[d] += a0 * b_val;
                                    c1[d] += a1 * b_val;
                                    c2[d] += a2 * b_val;
                                    c3[d] += a3 * b_val;
                                }
                            }

                            // Write back / accumulate to C
                            for d in 0..NR {
                                c_chunk[c_row0 + j + d] += c0[d];
                                c_chunk[c_row1 + j + d] += c1[d];
                                c_chunk[c_row2 + j + d] += c2[d];
                                c_chunk[c_row3 + j + d] += c3[d];
                            }

                            j += NR;
                        }

                        // Remainder columns in N (< 16)
                        while j < n_end {
                            for kk in k_block..k_end {
                                let a0 = a_row0[kk];
                                let a1 = a_row1[kk];
                                let a2 = a_row2[kk];
                                let a3 = a_row3[kk];
                                let b_val = b[kk * n + j];

                                c_chunk[c_row0 + j] += a0 * b_val;
                                c_chunk[c_row1 + j] += a1 * b_val;
                                c_chunk[c_row2 + j] += a2 * b_val;
                                c_chunk[c_row3 + j] += a3 * b_val;
                            }
                            j += 1;
                        }

                        i += MR;
                    }

                    // Remainder rows in M (< 4)
                    while i < m_len {
                        let c_row = &mut c_chunk[i * n..];
                        let a_row = &a_chunk[i * k..];

                        for kk in k_block..k_end {
                            let a_ik = a_row[kk];
                            let b_row = &b[kk * n..];

                            let mut j = n_block;
                            while j + 8 <= n_end {
                                c_row[j] += a_ik * b_row[j];
                                c_row[j + 1] += a_ik * b_row[j + 1];
                                c_row[j + 2] += a_ik * b_row[j + 2];
                                c_row[j + 3] += a_ik * b_row[j + 3];
                                c_row[j + 4] += a_ik * b_row[j + 4];
                                c_row[j + 5] += a_ik * b_row[j + 5];
                                c_row[j + 6] += a_ik * b_row[j + 6];
                                c_row[j + 7] += a_ik * b_row[j + 7];
                                j += 8;
                            }
                            while j < n_end {
                                c_row[j] += a_ik * b_row[j];
                                j += 1;
                            }
                        }
                        i += 1;
                    }
                }
            }
        });
}

/// Fast-path matrix-vector product for M = 1 (parallelized across columns of B).
pub fn gemv_m1_parallel(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = n.div_ceil(num_threads).max(64);

    c.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_sub)| {
            let j_start = chunk_idx * chunk_size;
            let j_len = c_sub.len();

            for kk in 0..k {
                let a_val = a[kk];
                let b_row = &b[kk * n + j_start..];

                let rem = j_len % 16;
                let unroll_len = j_len - rem;
                let mut j = 0;
                while j < unroll_len {
                    c_sub[j] += a_val * b_row[j];
                    c_sub[j + 1] += a_val * b_row[j + 1];
                    c_sub[j + 2] += a_val * b_row[j + 2];
                    c_sub[j + 3] += a_val * b_row[j + 3];
                    c_sub[j + 4] += a_val * b_row[j + 4];
                    c_sub[j + 5] += a_val * b_row[j + 5];
                    c_sub[j + 6] += a_val * b_row[j + 6];
                    c_sub[j + 7] += a_val * b_row[j + 7];
                    c_sub[j + 8] += a_val * b_row[j + 8];
                    c_sub[j + 9] += a_val * b_row[j + 9];
                    c_sub[j + 10] += a_val * b_row[j + 10];
                    c_sub[j + 11] += a_val * b_row[j + 11];
                    c_sub[j + 12] += a_val * b_row[j + 12];
                    c_sub[j + 13] += a_val * b_row[j + 13];
                    c_sub[j + 14] += a_val * b_row[j + 14];
                    c_sub[j + 15] += a_val * b_row[j + 15];
                    j += 16;
                }
                while j < j_len {
                    c_sub[j] += a_val * b_row[j];
                    j += 1;
                }
            }
        });
}

/// Computes matrix multiplication C = A * B^T directly without allocating a transposed B buffer.
/// A is [M, K] row-major, B is [N, K] row-major, C is [M, N] row-major.
pub fn gemm_2d_a_bt_contiguous(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    let num_threads = rayon::current_num_threads();
    let chunk_size = m.div_ceil(num_threads).max(16);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = (c_chunk.len() / n).min(m.saturating_sub(m_start));

            for i in 0..m_len {
                let a_row = &a[(m_start + i) * k..(m_start + i + 1) * k];
                let c_row = &mut c_chunk[i * n..(i + 1) * n];

                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];

                    // Vectorized dot product between a_row and b_row (both unit-stride!)
                    let mut sum0 = 0.0f32;
                    let mut sum1 = 0.0f32;
                    let mut sum2 = 0.0f32;
                    let mut sum3 = 0.0f32;

                    let rem = k % 16;
                    let unroll_len = k - rem;
                    let mut kk = 0;
                    while kk < unroll_len {
                        sum0 += a_row[kk] * b_row[kk];
                        sum1 += a_row[kk + 1] * b_row[kk + 1];
                        sum2 += a_row[kk + 2] * b_row[kk + 2];
                        sum3 += a_row[kk + 3] * b_row[kk + 3];

                        sum0 += a_row[kk + 4] * b_row[kk + 4];
                        sum1 += a_row[kk + 5] * b_row[kk + 5];
                        sum2 += a_row[kk + 6] * b_row[kk + 6];
                        sum3 += a_row[kk + 7] * b_row[kk + 7];

                        sum0 += a_row[kk + 8] * b_row[kk + 8];
                        sum1 += a_row[kk + 9] * b_row[kk + 9];
                        sum2 += a_row[kk + 10] * b_row[kk + 10];
                        sum3 += a_row[kk + 11] * b_row[kk + 11];

                        sum0 += a_row[kk + 12] * b_row[kk + 12];
                        sum1 += a_row[kk + 13] * b_row[kk + 13];
                        sum2 += a_row[kk + 14] * b_row[kk + 14];
                        sum3 += a_row[kk + 15] * b_row[kk + 15];
                        kk += 16;
                    }
                    while kk < k {
                        sum0 += a_row[kk] * b_row[kk];
                        kk += 1;
                    }

                    c_row[j] += (sum0 + sum1) + (sum2 + sum3);
                }
            }
        });
}

/// Computes matrix multiplication between two tensors, supporting arbitrary batch dimensions and broadcasting.
pub fn matmul(a: &RawTensor, b: &RawTensor) -> Result<RawTensor> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(EngineError::IncompatibleShapes {
            op: "matmul",
            shapes: vec![shape_a.to_vec(), shape_b.to_vec()],
        });
    }

    let m = shape_a[shape_a.len() - 2];
    let k_a = shape_a[shape_a.len() - 1];
    let k_b = shape_b[shape_b.len() - 2];
    let n = shape_b[shape_b.len() - 1];

    if k_a != k_b {
        return Err(EngineError::IncompatibleShapes {
            op: "matmul (inner dimensions mismatch)",
            shapes: vec![shape_a.to_vec(), shape_b.to_vec()],
        });
    }

    let k = k_a;
    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];

    let batch_out = broadcast_shapes(batch_a, batch_b)?;
    let mut out_shape = batch_out.clone();
    out_shape.push(m);
    out_shape.push(n);

    let a_contig = a.to_contiguous();
    let b_contig = b.to_contiguous();

    let a_slice = a_contig.as_slice();
    let b_slice = b_contig.as_slice();

    let total_batch_elements = numel(&batch_out);
    let mut out_data = vec![0.0; total_batch_elements * m * n];

    if total_batch_elements == 1 {
        // Direct 2D multiplication
        gemm_2d_contiguous(m, k, n, a_slice, b_slice, &mut out_data);
    } else {
        // Batched GEMM parallelized over batch index
        let a_batch_numel = numel(batch_a);
        let b_batch_numel = numel(batch_b);
        let a_strides = compute_c_contiguous_strides(batch_a);
        let b_strides = compute_c_contiguous_strides(batch_b);

        let matrix_size_a = m * k;
        let matrix_size_b = k * n;
        let matrix_size_c = m * n;

        out_data
            .par_chunks_mut(matrix_size_c)
            .enumerate()
            .for_each(|(b_idx, c_slice)| {
                // Compute batch coordinate in output shape
                let mut multi = vec![0; batch_out.len()];
                let mut rem = b_idx;
                for i in (0..batch_out.len()).rev() {
                    let dim = batch_out[i];
                    if dim > 0 {
                        multi[i] = rem % dim;
                        rem /= dim;
                    }
                }

                // Compute corresponding batch offset in A
                let mut a_offset = 0;
                if a_batch_numel > 1 {
                    let offset_dim = batch_out.len() - batch_a.len();
                    for (i, &dim_size) in batch_a.iter().enumerate() {
                        let coord = if dim_size == 1 {
                            0
                        } else {
                            multi[i + offset_dim]
                        };
                        a_offset += coord * a_strides[i];
                    }
                }

                // Compute corresponding batch offset in B
                let mut b_offset = 0;
                if b_batch_numel > 1 {
                    let offset_dim = batch_out.len() - batch_b.len();
                    for (i, &dim_size) in batch_b.iter().enumerate() {
                        let coord = if dim_size == 1 {
                            0
                        } else {
                            multi[i + offset_dim]
                        };
                        b_offset += coord * b_strides[i];
                    }
                }

                let sub_a = &a_slice[a_offset * matrix_size_a..(a_offset + 1) * matrix_size_a];
                let sub_b = &b_slice[b_offset * matrix_size_b..(b_offset + 1) * matrix_size_b];

                gemm_2d_serial(m, k, n, sub_a, sub_b, c_slice);
            });
    }

    Ok(RawTensor::from_vec(out_data, out_shape))
}

fn gemm_2d_serial(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        let a_row = &a[i * k..(i + 1) * k];

        for kk in 0..k {
            let a_ik = a_row[kk];
            let b_row = &b[kk * n..(kk + 1) * n];

            let rem = n % 16;
            let unroll_len = n - rem;

            let mut j = 0;
            while j < unroll_len {
                c_row[j] += a_ik * b_row[j];
                c_row[j + 1] += a_ik * b_row[j + 1];
                c_row[j + 2] += a_ik * b_row[j + 2];
                c_row[j + 3] += a_ik * b_row[j + 3];
                c_row[j + 4] += a_ik * b_row[j + 4];
                c_row[j + 5] += a_ik * b_row[j + 5];
                c_row[j + 6] += a_ik * b_row[j + 6];
                c_row[j + 7] += a_ik * b_row[j + 7];
                c_row[j + 8] += a_ik * b_row[j + 8];
                c_row[j + 9] += a_ik * b_row[j + 9];
                c_row[j + 10] += a_ik * b_row[j + 10];
                c_row[j + 11] += a_ik * b_row[j + 11];
                c_row[j + 12] += a_ik * b_row[j + 12];
                c_row[j + 13] += a_ik * b_row[j + 13];
                c_row[j + 14] += a_ik * b_row[j + 14];
                c_row[j + 15] += a_ik * b_row[j + 15];
                j += 16;
            }
            while j < n {
                c_row[j] += a_ik * b_row[j];
                j += 1;
            }
        }
    }
}

/// Multiplies A by transposed B (C = A * B^T) directly without allocating a transposed B buffer.
/// A has shape [..., M, K], B has shape [..., N, K]. Output has shape [..., M, N].
pub fn matmul_transposed_b(a: &RawTensor, b: &RawTensor) -> Result<RawTensor> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(EngineError::IncompatibleShapes {
            op: "matmul_transposed_b",
            shapes: vec![shape_a.to_vec(), shape_b.to_vec()],
        });
    }

    let m = shape_a[shape_a.len() - 2];
    let k_a = shape_a[shape_a.len() - 1];
    let n = shape_b[shape_b.len() - 2];
    let k_b = shape_b[shape_b.len() - 1];

    if k_a != k_b {
        return Err(EngineError::IncompatibleShapes {
            op: "matmul_transposed_b (inner dimensions mismatch)",
            shapes: vec![shape_a.to_vec(), shape_b.to_vec()],
        });
    }

    let k = k_a;
    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];

    let batch_out = broadcast_shapes(batch_a, batch_b)?;
    let mut out_shape = batch_out.clone();
    out_shape.push(m);
    out_shape.push(n);

    let a_contig = a.to_contiguous();
    let b_contig = b.to_contiguous();

    let a_slice = a_contig.as_slice();
    let b_slice = b_contig.as_slice();

    let total_batch_elements = numel(&batch_out);
    let mut out_data = vec![0.0; total_batch_elements * m * n];

    if total_batch_elements == 1 {
        gemm_2d_a_bt_contiguous(m, k, n, a_slice, b_slice, &mut out_data);
    } else {
        let b_strides = compute_c_contiguous_strides(batch_b);
        let a_strides = compute_c_contiguous_strides(batch_a);
        let a_batch_numel = numel(batch_a);
        let b_batch_numel = numel(batch_b);

        let matrix_size_a = m * k;
        let matrix_size_b = n * k;
        let matrix_size_c = m * n;

        out_data
            .par_chunks_mut(matrix_size_c)
            .enumerate()
            .for_each(|(b_idx, c_slice)| {
                let mut multi = vec![0; batch_out.len()];
                let mut rem = b_idx;
                for i in (0..batch_out.len()).rev() {
                    let dim = batch_out[i];
                    if dim > 0 {
                        multi[i] = rem % dim;
                        rem /= dim;
                    }
                }

                let mut a_offset = 0;
                if a_batch_numel > 1 {
                    let offset_dim = batch_out.len() - batch_a.len();
                    for (i, &dim_size) in batch_a.iter().enumerate() {
                        let coord = if dim_size == 1 {
                            0
                        } else {
                            multi[i + offset_dim]
                        };
                        a_offset += coord * a_strides[i];
                    }
                }

                let mut b_offset = 0;
                if b_batch_numel > 1 {
                    let offset_dim = batch_out.len() - batch_b.len();
                    for (i, &dim_size) in batch_b.iter().enumerate() {
                        let coord = if dim_size == 1 {
                            0
                        } else {
                            multi[i + offset_dim]
                        };
                        b_offset += coord * b_strides[i];
                    }
                }

                let sub_a = &a_slice[a_offset * matrix_size_a..(a_offset + 1) * matrix_size_a];
                let sub_b = &b_slice[b_offset * matrix_size_b..(b_offset + 1) * matrix_size_b];

                gemm_2d_a_bt_contiguous(m, k, n, sub_a, sub_b, c_slice);
            });
    }

    Ok(RawTensor::from_vec(out_data, out_shape))
}
