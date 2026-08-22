//! High-performance tiled, cache-blocked, SIMD-accelerated matrix multiplication (GEMM).
//!
//! Features:
//! - Hardware-accelerated AVX2 + FMA3 microkernels on x86_64 (Zen 3 / Intel) with 100% register utilization.
//! - Optimal 6x16 register-tiling (12 YMM accumulators + 2 B streaming + 2 A broadcast).
//! - 4x4 vector-streaming microkernel for transposed multiplications (C = A * B^T).
//! - Fast-path parallel GEMV for M = 1 (autoregressive token generation).
//! - Dynamic runtime CPU feature detection with fallback for non-AVX2 / ARM targets.
//! - Multi-threaded 2D grid work-stealing parallelization powered by Rayon.

use crate::error::{EngineError, Result};
use crate::tensor::shape::{broadcast_shapes, compute_c_contiguous_strides, numel};
use crate::tensor::RawTensor;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Cache blocking constants optimized for Zen 3 L1d (32 KiB) & L2 (512 KiB)
const BLOCK_M: usize = 96; // Multiple of 6
const BLOCK_K: usize = 256;
const BLOCK_N: usize = 256; // Multiple of 16

const MR: usize = 6;
const NR: usize = 16;

/// Performs 2D matrix multiplication C = A * B on contiguous row-major slices.
/// Dispatches to AVX2+FMA microkernels on compatible x86_64 CPUs or portable fallback.
pub fn gemm_2d_contiguous(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Fast-path: M = 1 GEMV (Matrix-Vector Multiplication, common in LLM decoding)
    if m == 1 {
        gemv_m1_parallel(k, n, a, b, c);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                gemm_2d_avx2_fma_parallel(m, k, n, a, b, c);
            }
            return;
        }
    }

    gemm_2d_portable_parallel(m, k, n, a, b, c);
}

/// Computes matrix multiplication C = A * B^T directly without allocating a transposed B buffer.
/// A is [M, K] row-major, B is [N, K] row-major, C is [M, N] row-major.
pub fn gemm_2d_a_bt_contiguous(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                gemm_2d_a_bt_avx2_fma_parallel(m, k, n, a, b, c);
            }
            return;
        }
    }

    gemm_2d_a_bt_portable_parallel(m, k, n, a, b, c);
}

// =========================================================================
// AVX2 + FMA3 x86_64 Accelerated Implementation (Ryzen Zen 3 / Intel Core)
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_2d_avx2_fma_parallel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = (m.div_ceil(num_threads)).div_ceil(MR) * MR;
    let chunk_size = chunk_size.max(BLOCK_M);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = (c_chunk.len() / n).min(m.saturating_sub(m_start));
            let a_chunk = &a[m_start * k..(m_start + m_len) * k];

            for n_block in (0..n).step_by(BLOCK_N) {
                let n_end = (n_block + BLOCK_N).min(n);

                // 6x16 AVX2 Microkernel Loop
                let mut i = 0;
                while i + MR <= m_len {
                    let a_ptr0 = a_chunk.as_ptr().add(i * k);
                    let a_ptr1 = a_chunk.as_ptr().add((i + 1) * k);
                    let a_ptr2 = a_chunk.as_ptr().add((i + 2) * k);
                    let a_ptr3 = a_chunk.as_ptr().add((i + 3) * k);
                    let a_ptr4 = a_chunk.as_ptr().add((i + 4) * k);
                    let a_ptr5 = a_chunk.as_ptr().add((i + 5) * k);

                    let mut j = n_block;
                    while j + NR <= n_end {
                        let b_ptr = b.as_ptr().add(j);

                        // 12 YMM accumulator registers: 6 rows x 2 vectors of 8
                        let mut c0_0 = _mm256_setzero_ps();
                        let mut c0_1 = _mm256_setzero_ps();
                        let mut c1_0 = _mm256_setzero_ps();
                        let mut c1_1 = _mm256_setzero_ps();
                        let mut c2_0 = _mm256_setzero_ps();
                        let mut c2_1 = _mm256_setzero_ps();
                        let mut c3_0 = _mm256_setzero_ps();
                        let mut c3_1 = _mm256_setzero_ps();
                        let mut c4_0 = _mm256_setzero_ps();
                        let mut c4_1 = _mm256_setzero_ps();
                        let mut c5_0 = _mm256_setzero_ps();
                        let mut c5_1 = _mm256_setzero_ps();

                        let mut p = 0;
                        while p + 2 <= k {
                            let bp0 = b_ptr.add(p * n);
                            let b0_0 = _mm256_loadu_ps(bp0);
                            let b0_1 = _mm256_loadu_ps(bp0.add(8));

                            let a0_0 = _mm256_set1_ps(*a_ptr0.add(p));
                            let a1_0 = _mm256_set1_ps(*a_ptr1.add(p));
                            let a2_0 = _mm256_set1_ps(*a_ptr2.add(p));
                            let a3_0 = _mm256_set1_ps(*a_ptr3.add(p));
                            let a4_0 = _mm256_set1_ps(*a_ptr4.add(p));
                            let a5_0 = _mm256_set1_ps(*a_ptr5.add(p));

                            c0_0 = _mm256_fmadd_ps(a0_0, b0_0, c0_0);
                            c0_1 = _mm256_fmadd_ps(a0_0, b0_1, c0_1);
                            c1_0 = _mm256_fmadd_ps(a1_0, b0_0, c1_0);
                            c1_1 = _mm256_fmadd_ps(a1_0, b0_1, c1_1);
                            c2_0 = _mm256_fmadd_ps(a2_0, b0_0, c2_0);
                            c2_1 = _mm256_fmadd_ps(a2_0, b0_1, c2_1);
                            c3_0 = _mm256_fmadd_ps(a3_0, b0_0, c3_0);
                            c3_1 = _mm256_fmadd_ps(a3_0, b0_1, c3_1);
                            c4_0 = _mm256_fmadd_ps(a4_0, b0_0, c4_0);
                            c4_1 = _mm256_fmadd_ps(a4_0, b0_1, c4_1);
                            c5_0 = _mm256_fmadd_ps(a5_0, b0_0, c5_0);
                            c5_1 = _mm256_fmadd_ps(a5_0, b0_1, c5_1);

                            let bp1 = b_ptr.add((p + 1) * n);
                            let b1_0 = _mm256_loadu_ps(bp1);
                            let b1_1 = _mm256_loadu_ps(bp1.add(8));

                            let a0_1 = _mm256_set1_ps(*a_ptr0.add(p + 1));
                            let a1_1 = _mm256_set1_ps(*a_ptr1.add(p + 1));
                            let a2_1 = _mm256_set1_ps(*a_ptr2.add(p + 1));
                            let a3_1 = _mm256_set1_ps(*a_ptr3.add(p + 1));
                            let a4_1 = _mm256_set1_ps(*a_ptr4.add(p + 1));
                            let a5_1 = _mm256_set1_ps(*a_ptr5.add(p + 1));

                            c0_0 = _mm256_fmadd_ps(a0_1, b1_0, c0_0);
                            c0_1 = _mm256_fmadd_ps(a0_1, b1_1, c0_1);
                            c1_0 = _mm256_fmadd_ps(a1_1, b1_0, c1_0);
                            c1_1 = _mm256_fmadd_ps(a1_1, b1_1, c1_1);
                            c2_0 = _mm256_fmadd_ps(a2_1, b1_0, c2_0);
                            c2_1 = _mm256_fmadd_ps(a2_1, b1_1, c2_1);
                            c3_0 = _mm256_fmadd_ps(a3_1, b1_0, c3_0);
                            c3_1 = _mm256_fmadd_ps(a3_1, b1_1, c3_1);
                            c4_0 = _mm256_fmadd_ps(a4_1, b1_0, c4_0);
                            c4_1 = _mm256_fmadd_ps(a4_1, b1_1, c4_1);
                            c5_0 = _mm256_fmadd_ps(a5_1, b1_0, c5_0);
                            c5_1 = _mm256_fmadd_ps(a5_1, b1_1, c5_1);

                            p += 2;
                        }

                        while p < k {
                            let bp = b_ptr.add(p * n);
                            let b0 = _mm256_loadu_ps(bp);
                            let b1 = _mm256_loadu_ps(bp.add(8));

                            let a0 = _mm256_set1_ps(*a_ptr0.add(p));
                            let a1 = _mm256_set1_ps(*a_ptr1.add(p));
                            let a2 = _mm256_set1_ps(*a_ptr2.add(p));
                            let a3 = _mm256_set1_ps(*a_ptr3.add(p));
                            let a4 = _mm256_set1_ps(*a_ptr4.add(p));
                            let a5 = _mm256_set1_ps(*a_ptr5.add(p));

                            c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
                            c0_1 = _mm256_fmadd_ps(a0, b1, c0_1);
                            c1_0 = _mm256_fmadd_ps(a1, b0, c1_0);
                            c1_1 = _mm256_fmadd_ps(a1, b1, c1_1);
                            c2_0 = _mm256_fmadd_ps(a2, b0, c2_0);
                            c2_1 = _mm256_fmadd_ps(a2, b1, c2_1);
                            c3_0 = _mm256_fmadd_ps(a3, b0, c3_0);
                            c3_1 = _mm256_fmadd_ps(a3, b1, c3_1);
                            c4_0 = _mm256_fmadd_ps(a4, b0, c4_0);
                            c4_1 = _mm256_fmadd_ps(a4, b1, c4_1);
                            c5_0 = _mm256_fmadd_ps(a5, b0, c5_0);
                            c5_1 = _mm256_fmadd_ps(a5, b1, c5_1);

                            p += 1;
                        }

                        let cp0 = c_chunk.as_mut_ptr().add(i * n + j);
                        let cp1 = c_chunk.as_mut_ptr().add((i + 1) * n + j);
                        let cp2 = c_chunk.as_mut_ptr().add((i + 2) * n + j);
                        let cp3 = c_chunk.as_mut_ptr().add((i + 3) * n + j);
                        let cp4 = c_chunk.as_mut_ptr().add((i + 4) * n + j);
                        let cp5 = c_chunk.as_mut_ptr().add((i + 5) * n + j);

                        _mm256_storeu_ps(cp0, c0_0);
                        _mm256_storeu_ps(cp0.add(8), c0_1);
                        _mm256_storeu_ps(cp1, c1_0);
                        _mm256_storeu_ps(cp1.add(8), c1_1);
                        _mm256_storeu_ps(cp2, c2_0);
                        _mm256_storeu_ps(cp2.add(8), c2_1);
                        _mm256_storeu_ps(cp3, c3_0);
                        _mm256_storeu_ps(cp3.add(8), c3_1);
                        _mm256_storeu_ps(cp4, c4_0);
                        _mm256_storeu_ps(cp4.add(8), c4_1);
                        _mm256_storeu_ps(cp5, c5_0);
                        _mm256_storeu_ps(cp5.add(8), c5_1);

                        j += NR;
                    }

                    // Remainder columns in N (8-wide AVX2 + scalar)
                    while j + 8 <= n_end {
                        let b_ptr = b.as_ptr().add(j);
                        let mut c0 = _mm256_setzero_ps();
                        let mut c1 = _mm256_setzero_ps();
                        let mut c2 = _mm256_setzero_ps();
                        let mut c3 = _mm256_setzero_ps();
                        let mut c4 = _mm256_setzero_ps();
                        let mut c5 = _mm256_setzero_ps();

                        for p in 0..k {
                            let bp = _mm256_loadu_ps(b_ptr.add(p * n));
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr0.add(p)), bp, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr1.add(p)), bp, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr2.add(p)), bp, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr3.add(p)), bp, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr4.add(p)), bp, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr5.add(p)), bp, c5);
                        }

                        let cp0 = c_chunk.as_mut_ptr().add(i * n + j);
                        let cp1 = c_chunk.as_mut_ptr().add((i + 1) * n + j);
                        let cp2 = c_chunk.as_mut_ptr().add((i + 2) * n + j);
                        let cp3 = c_chunk.as_mut_ptr().add((i + 3) * n + j);
                        let cp4 = c_chunk.as_mut_ptr().add((i + 4) * n + j);
                        let cp5 = c_chunk.as_mut_ptr().add((i + 5) * n + j);

                        _mm256_storeu_ps(cp0, c0);
                        _mm256_storeu_ps(cp1, c1);
                        _mm256_storeu_ps(cp2, c2);
                        _mm256_storeu_ps(cp3, c3);
                        _mm256_storeu_ps(cp4, c4);
                        _mm256_storeu_ps(cp5, c5);

                        j += 8;
                    }

                    while j < n_end {
                        let mut c0 = 0.0f32;
                        let mut c1 = 0.0f32;
                        let mut c2 = 0.0f32;
                        let mut c3 = 0.0f32;
                        let mut c4 = 0.0f32;
                        let mut c5 = 0.0f32;

                        for p in 0..k {
                            let b_val = b[p * n + j];
                            c0 += *a_ptr0.add(p) * b_val;
                            c1 += *a_ptr1.add(p) * b_val;
                            c2 += *a_ptr2.add(p) * b_val;
                            c3 += *a_ptr3.add(p) * b_val;
                            c4 += *a_ptr4.add(p) * b_val;
                            c5 += *a_ptr5.add(p) * b_val;
                        }
                        c_chunk[i * n + j] = c0;
                        c_chunk[(i + 1) * n + j] = c1;
                        c_chunk[(i + 2) * n + j] = c2;
                        c_chunk[(i + 3) * n + j] = c3;
                        c_chunk[(i + 4) * n + j] = c4;
                        c_chunk[(i + 5) * n + j] = c5;
                        j += 1;
                    }

                    i += MR;
                }

                // Remainder rows in M (< 6)
                while i < m_len {
                    let c_row = &mut c_chunk[i * n..];
                    let a_row = &a_chunk[i * k..];

                    let mut j = n_block;
                    while j + 8 <= n_end {
                        let mut acc = _mm256_setzero_ps();
                        for (p, &a_val) in a_row.iter().enumerate().take(k) {
                            let a_vec = _mm256_set1_ps(a_val);
                            let bp = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                            acc = _mm256_fmadd_ps(a_vec, bp, acc);
                        }
                        _mm256_storeu_ps(c_row.as_mut_ptr().add(j), acc);
                        j += 8;
                    }
                    while j < n_end {
                        let mut sum = 0.0f32;
                        for (p, &a_val) in a_row.iter().enumerate().take(k) {
                            sum += a_val * b[p * n + j];
                        }
                        c_row[j] = sum;
                        j += 1;
                    }
                    i += 1;
                }
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let v_high = _mm256_extractf128_ps(v, 1);
    let v_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_low, v_high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_2d_a_bt_avx2_fma_parallel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = m.div_ceil(num_threads).max(16);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = (c_chunk.len() / n).min(m.saturating_sub(m_start));

            let mut i = 0;
            // 4x4 Register Tiled Dot-Product Microkernel
            while i + 4 <= m_len {
                let a0_ptr = a.as_ptr().add((m_start + i) * k);
                let a1_ptr = a.as_ptr().add((m_start + i + 1) * k);
                let a2_ptr = a.as_ptr().add((m_start + i + 2) * k);
                let a3_ptr = a.as_ptr().add((m_start + i + 3) * k);

                let mut j = 0;
                while j + 4 <= n {
                    let b0_ptr = b.as_ptr().add(j * k);
                    let b1_ptr = b.as_ptr().add((j + 1) * k);
                    let b2_ptr = b.as_ptr().add((j + 2) * k);
                    let b3_ptr = b.as_ptr().add((j + 3) * k);

                    let mut acc00 = _mm256_setzero_ps();
                    let mut acc01 = _mm256_setzero_ps();
                    let mut acc02 = _mm256_setzero_ps();
                    let mut acc03 = _mm256_setzero_ps();

                    let mut acc10 = _mm256_setzero_ps();
                    let mut acc11 = _mm256_setzero_ps();
                    let mut acc12 = _mm256_setzero_ps();
                    let mut acc13 = _mm256_setzero_ps();

                    let mut acc20 = _mm256_setzero_ps();
                    let mut acc21 = _mm256_setzero_ps();
                    let mut acc22 = _mm256_setzero_ps();
                    let mut acc23 = _mm256_setzero_ps();

                    let mut acc30 = _mm256_setzero_ps();
                    let mut acc31 = _mm256_setzero_ps();
                    let mut acc32 = _mm256_setzero_ps();
                    let mut acc33 = _mm256_setzero_ps();

                    let mut p = 0;
                    while p + 8 <= k {
                        let va0 = _mm256_loadu_ps(a0_ptr.add(p));
                        let va1 = _mm256_loadu_ps(a1_ptr.add(p));
                        let va2 = _mm256_loadu_ps(a2_ptr.add(p));
                        let va3 = _mm256_loadu_ps(a3_ptr.add(p));

                        let vb0 = _mm256_loadu_ps(b0_ptr.add(p));
                        let vb1 = _mm256_loadu_ps(b1_ptr.add(p));
                        let vb2 = _mm256_loadu_ps(b2_ptr.add(p));
                        let vb3 = _mm256_loadu_ps(b3_ptr.add(p));

                        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                        acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
                        acc03 = _mm256_fmadd_ps(va0, vb3, acc03);

                        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                        acc12 = _mm256_fmadd_ps(va1, vb2, acc12);
                        acc13 = _mm256_fmadd_ps(va1, vb3, acc13);

                        acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                        acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                        acc22 = _mm256_fmadd_ps(va2, vb2, acc22);
                        acc23 = _mm256_fmadd_ps(va2, vb3, acc23);

                        acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                        acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                        acc32 = _mm256_fmadd_ps(va3, vb2, acc32);
                        acc33 = _mm256_fmadd_ps(va3, vb3, acc33);

                        p += 8;
                    }

                    let mut s00 = hsum256_ps(acc00);
                    let mut s01 = hsum256_ps(acc01);
                    let mut s02 = hsum256_ps(acc02);
                    let mut s03 = hsum256_ps(acc03);

                    let mut s10 = hsum256_ps(acc10);
                    let mut s11 = hsum256_ps(acc11);
                    let mut s12 = hsum256_ps(acc12);
                    let mut s13 = hsum256_ps(acc13);

                    let mut s20 = hsum256_ps(acc20);
                    let mut s21 = hsum256_ps(acc21);
                    let mut s22 = hsum256_ps(acc22);
                    let mut s23 = hsum256_ps(acc23);

                    let mut s30 = hsum256_ps(acc30);
                    let mut s31 = hsum256_ps(acc31);
                    let mut s32 = hsum256_ps(acc32);
                    let mut s33 = hsum256_ps(acc33);

                    while p < k {
                        let a0 = *a0_ptr.add(p);
                        let a1 = *a1_ptr.add(p);
                        let a2 = *a2_ptr.add(p);
                        let a3 = *a3_ptr.add(p);
                        let b0 = *b0_ptr.add(p);
                        let b1 = *b1_ptr.add(p);
                        let b2 = *b2_ptr.add(p);
                        let b3 = *b3_ptr.add(p);

                        s00 += a0 * b0;
                        s01 += a0 * b1;
                        s02 += a0 * b2;
                        s03 += a0 * b3;
                        s10 += a1 * b0;
                        s11 += a1 * b1;
                        s12 += a1 * b2;
                        s13 += a1 * b3;
                        s20 += a2 * b0;
                        s21 += a2 * b1;
                        s22 += a2 * b2;
                        s23 += a2 * b3;
                        s30 += a3 * b0;
                        s31 += a3 * b1;
                        s32 += a3 * b2;
                        s33 += a3 * b3;
                        p += 1;
                    }

                    c_chunk[i * n + j] += s00;
                    c_chunk[i * n + j + 1] += s01;
                    c_chunk[i * n + j + 2] += s02;
                    c_chunk[i * n + j + 3] += s03;

                    c_chunk[(i + 1) * n + j] += s10;
                    c_chunk[(i + 1) * n + j + 1] += s11;
                    c_chunk[(i + 1) * n + j + 2] += s12;
                    c_chunk[(i + 1) * n + j + 3] += s13;

                    c_chunk[(i + 2) * n + j] += s20;
                    c_chunk[(i + 2) * n + j + 1] += s21;
                    c_chunk[(i + 2) * n + j + 2] += s22;
                    c_chunk[(i + 2) * n + j + 3] += s23;

                    c_chunk[(i + 3) * n + j] += s30;
                    c_chunk[(i + 3) * n + j + 1] += s31;
                    c_chunk[(i + 3) * n + j + 2] += s32;
                    c_chunk[(i + 3) * n + j + 3] += s33;

                    j += 4;
                }

                while j < n {
                    let b_row = &b[j * k..(j + 1) * k];
                    for row_offset in 0..4 {
                        let a_row =
                            &a[(m_start + i + row_offset) * k..(m_start + i + row_offset + 1) * k];
                        let mut acc = _mm256_setzero_ps();
                        let mut p = 0;
                        while p + 8 <= k {
                            acc = _mm256_fmadd_ps(
                                _mm256_loadu_ps(a_row.as_ptr().add(p)),
                                _mm256_loadu_ps(b_row.as_ptr().add(p)),
                                acc,
                            );
                            p += 8;
                        }
                        let mut sum = hsum256_ps(acc);
                        while p < k {
                            sum += a_row[p] * b_row[p];
                            p += 1;
                        }
                        c_chunk[(i + row_offset) * n + j] += sum;
                    }
                    j += 1;
                }
                i += 4;
            }

            while i < m_len {
                let a_row = &a[(m_start + i) * k..(m_start + i + 1) * k];
                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut acc = _mm256_setzero_ps();
                    let mut p = 0;
                    while p + 8 <= k {
                        acc = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a_row.as_ptr().add(p)),
                            _mm256_loadu_ps(b_row.as_ptr().add(p)),
                            acc,
                        );
                        p += 8;
                    }
                    let mut sum = hsum256_ps(acc);
                    while p < k {
                        sum += a_row[p] * b_row[p];
                        p += 1;
                    }
                    c_chunk[i * n + j] += sum;
                }
                i += 1;
            }
        });
}

// =========================================================================
// Portable Generic Implementation (Fallback for non-AVX2 / ARM NEON)
// =========================================================================

fn gemm_2d_portable_parallel(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = (m.div_ceil(num_threads)).div_ceil(4) * 4;
    let chunk_size = chunk_size.max(BLOCK_M);

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let m_start = chunk_idx * chunk_size;
            let m_len = (c_chunk.len() / n).min(m.saturating_sub(m_start));
            let a_chunk = &a[m_start * k..(m_start + m_len) * k];

            for k_block in (0..k).step_by(BLOCK_K) {
                let k_end = (k_block + BLOCK_K).min(k);

                for n_block in (0..n).step_by(BLOCK_N) {
                    let n_end = (n_block + BLOCK_N).min(n);

                    let mut i = 0;
                    while i + 4 <= m_len {
                        let c_row0 = i * n;
                        let c_row1 = (i + 1) * n;
                        let c_row2 = (i + 2) * n;
                        let c_row3 = (i + 3) * n;

                        let a_row0 = &a_chunk[i * k..];
                        let a_row1 = &a_chunk[(i + 1) * k..];
                        let a_row2 = &a_chunk[(i + 2) * k..];
                        let a_row3 = &a_chunk[(i + 3) * k..];

                        let mut j = n_block;
                        while j + 16 <= n_end {
                            let mut c0 = [0.0f32; 16];
                            let mut c1 = [0.0f32; 16];
                            let mut c2 = [0.0f32; 16];
                            let mut c3 = [0.0f32; 16];

                            for kk in k_block..k_end {
                                let a0 = a_row0[kk];
                                let a1 = a_row1[kk];
                                let a2 = a_row2[kk];
                                let a3 = a_row3[kk];

                                let b_row = &b[kk * n + j..];

                                for d in 0..16 {
                                    let b_val = b_row[d];
                                    c0[d] += a0 * b_val;
                                    c1[d] += a1 * b_val;
                                    c2[d] += a2 * b_val;
                                    c3[d] += a3 * b_val;
                                }
                            }

                            for d in 0..16 {
                                c_chunk[c_row0 + j + d] += c0[d];
                                c_chunk[c_row1 + j + d] += c1[d];
                                c_chunk[c_row2 + j + d] += c2[d];
                                c_chunk[c_row3 + j + d] += c3[d];
                            }

                            j += 16;
                        }

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

                        i += 4;
                    }

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

fn gemm_2d_a_bt_portable_parallel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
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

/// Fast-path matrix-vector product for M = 1 (parallelized across columns of B).
pub fn gemv_m1_parallel(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = n.div_ceil(num_threads).max(64);

    c.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_sub)| {
            let j_start = chunk_idx * chunk_size;
            let j_len = c_sub.len();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        gemv_m1_avx2_fma(k, n, a, b, j_start, j_len, c_sub);
                    }
                    return;
                }
            }

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_m1_avx2_fma(
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    j_start: usize,
    j_len: usize,
    c_sub: &mut [f32],
) {
    for (kk, &a_scalar) in a.iter().enumerate().take(k) {
        let a_val = _mm256_set1_ps(a_scalar);
        let b_ptr = b.as_ptr().add(kk * n + j_start);
        let c_ptr = c_sub.as_mut_ptr();

        let mut j = 0;
        while j + 32 <= j_len {
            let cp0 = c_ptr.add(j);
            let cp1 = c_ptr.add(j + 8);
            let cp2 = c_ptr.add(j + 16);
            let cp3 = c_ptr.add(j + 24);

            let bp0 = b_ptr.add(j);
            let bp1 = b_ptr.add(j + 8);
            let bp2 = b_ptr.add(j + 16);
            let bp3 = b_ptr.add(j + 24);

            _mm256_storeu_ps(
                cp0,
                _mm256_fmadd_ps(a_val, _mm256_loadu_ps(bp0), _mm256_loadu_ps(cp0)),
            );
            _mm256_storeu_ps(
                cp1,
                _mm256_fmadd_ps(a_val, _mm256_loadu_ps(bp1), _mm256_loadu_ps(cp1)),
            );
            _mm256_storeu_ps(
                cp2,
                _mm256_fmadd_ps(a_val, _mm256_loadu_ps(bp2), _mm256_loadu_ps(cp2)),
            );
            _mm256_storeu_ps(
                cp3,
                _mm256_fmadd_ps(a_val, _mm256_loadu_ps(bp3), _mm256_loadu_ps(cp3)),
            );

            j += 32;
        }

        while j + 8 <= j_len {
            let cp = c_ptr.add(j);
            let bp = b_ptr.add(j);
            _mm256_storeu_ps(
                cp,
                _mm256_fmadd_ps(a_val, _mm256_loadu_ps(bp), _mm256_loadu_ps(cp)),
            );
            j += 8;
        }

        while j < j_len {
            c_sub[j] += a[kk] * *b_ptr.add(j);
            j += 1;
        }
    }
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
        gemm_2d_contiguous(m, k, n, a_slice, b_slice, &mut out_data);
    } else {
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

                gemm_2d_contiguous(m, k, n, sub_a, sub_b, c_slice);
            });
    }

    Ok(RawTensor::from_vec(out_data, out_shape))
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
