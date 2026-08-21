//! Tiled, cache-blocked, multi-threaded matrix multiplication (GEMM).

use crate::error::{EngineError, Result};
use crate::tensor::shape::{broadcast_shapes, compute_c_contiguous_strides, numel};
use crate::tensor::RawTensor;
use rayon::prelude::*;

const BLOCK_M: usize = 64;
const BLOCK_K: usize = 128;
const BLOCK_N: usize = 128;

/// Performs 2D matrix multiplication C = A * B on contiguous row-major slices.
/// Employs cache-blocked tiling and Rayon parallelization.
pub fn gemm_2d_contiguous(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Parallelize across row chunks of A and C
    let num_threads = rayon::current_num_threads();
    let chunk_size = m.div_ceil(num_threads).max(BLOCK_M);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = c_chunk.len() / n;
            let a_chunk = &a[m_start * k..(m_start + m_len) * k];

            // Tiled GEMM with unit-stride inner loops for SIMD autovectorization
            for k_block in (0..k).step_by(BLOCK_K) {
                let k_end = (k_block + BLOCK_K).min(k);

                for n_block in (0..n).step_by(BLOCK_N) {
                    let n_end = (n_block + BLOCK_N).min(n);

                    for i in 0..m_len {
                        let c_row = &mut c_chunk[i * n..];
                        let a_row = &a_chunk[i * k..];

                        for kk in k_block..k_end {
                            let a_ik = a_row[kk];
                            let b_row = &b[kk * n..];

                            // Inner loop: vectorized unit-stride FMA: C[i, j] += A[i, kk] * B[kk, j]
                            let c_slice = &mut c_row[n_block..n_end];
                            let b_slice = &b_row[n_block..n_end];

                            // Chunk of 8 for optimal register pipelining
                            let len = c_slice.len();
                            let rem = len % 8;
                            let unroll_len = len - rem;

                            let mut j = 0;
                            while j < unroll_len {
                                c_slice[j] += a_ik * b_slice[j];
                                c_slice[j + 1] += a_ik * b_slice[j + 1];
                                c_slice[j + 2] += a_ik * b_slice[j + 2];
                                c_slice[j + 3] += a_ik * b_slice[j + 3];
                                c_slice[j + 4] += a_ik * b_slice[j + 4];
                                c_slice[j + 5] += a_ik * b_slice[j + 5];
                                c_slice[j + 6] += a_ik * b_slice[j + 6];
                                c_slice[j + 7] += a_ik * b_slice[j + 7];
                                j += 8;
                            }
                            while j < len {
                                c_slice[j] += a_ik * b_slice[j];
                                j += 1;
                            }
                        }
                    }
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

                // Perform un-parallelized single matrix GEMM per batch item
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

            let len = n;
            let rem = len % 8;
            let unroll_len = len - rem;

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
                j += 8;
            }
            while j < len {
                c_row[j] += a_ik * b_row[j];
                j += 1;
            }
        }
    }
}
