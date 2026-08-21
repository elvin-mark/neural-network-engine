//! Property-based and fuzz-style verification tests for strided tensors, broadcasting, GEMM, and conv.

use neural_network_engine::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Reference naive matrix multiplication C = A * B where A is [M, K] and B is [K, N].
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Reference naive 2D convolution for [1, 1, H, W] with [1, 1, KH, KW], stride=1, pad=0.
fn naive_conv2d(
    input: &[f32],
    h: usize,
    w: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let out_h = h - kh + 1;
    let out_w = w - kw + 1;
    let mut out = vec![0.0f32; out_h * out_w];

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0f32;
            for r in 0..kh {
                for c in 0..kw {
                    sum += input[(oh + r) * w + (ow + c)] * kernel[r * kw + c];
                }
            }
            out[oh * out_w + ow] = sum;
        }
    }
    out
}

#[test]
fn test_property_random_gemm_vs_naive_reference() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test a wide variety of dimension configurations (including odd, prime, singletons, and rectangular)
    let test_dims = [
        (1, 1, 1),
        (1, 7, 1),
        (3, 1, 5),
        (13, 17, 19),
        (31, 23, 29),
        (47, 61, 37),
        (1, 64, 32),
        (64, 1, 64),
        (73, 53, 41),
    ];

    for (m, k, n) in test_dims {
        let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-2.0..2.0)).collect();

        let tensor_a = RawTensor::from_slice(&a_data, &[m, k]);
        let tensor_b = RawTensor::from_slice(&b_data, &[k, n]);

        let tensor_c = neural_network_engine::tensor::matmul::matmul(&tensor_a, &tensor_b).unwrap();
        let naive_c = naive_matmul(&a_data, &b_data, m, k, n);

        let actual = tensor_c.to_contiguous();
        let actual_slice = actual.as_slice();

        assert_eq!(actual_slice.len(), naive_c.len());
        for (idx, (&act, &exp)) in actual_slice.iter().zip(naive_c.iter()).enumerate() {
            let diff = (act - exp).abs();
            assert!(
                diff < 1e-4,
                "GEMM mismatch at dim ({}, {}, {}) index {}: actual {} vs expected {}",
                m,
                k,
                n,
                idx,
                act,
                exp
            );
        }
    }
}

#[test]
fn test_property_batched_gemm_vs_2d() {
    let mut rng = StdRng::seed_from_u64(12345);
    let batch_size = 4;
    let (m, k, n) = (8, 12, 16);

    let a_data: Vec<f32> = (0..batch_size * m * k)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let b_data: Vec<f32> = (0..batch_size * k * n)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let a_3d = RawTensor::from_slice(&a_data, &[batch_size, m, k]);
    let b_3d = RawTensor::from_slice(&b_data, &[batch_size, k, n]);

    let c_3d = neural_network_engine::tensor::matmul::matmul(&a_3d, &b_3d).unwrap();
    assert_eq!(c_3d.shape(), &[batch_size, m, n]);

    for b in 0..batch_size {
        let a_slice = a_3d.slice(0, b, b + 1).unwrap().reshape(&[m, k]).unwrap();
        let b_slice = b_3d.slice(0, b, b + 1).unwrap().reshape(&[k, n]).unwrap();
        let c_slice = c_3d.slice(0, b, b + 1).unwrap().reshape(&[m, n]).unwrap();

        let c_2d = neural_network_engine::tensor::matmul::matmul(&a_slice, &b_slice).unwrap();

        let c_3d_sub = c_slice.to_contiguous();
        let c_2d_contig = c_2d.to_contiguous();

        for (idx, (&act, &exp)) in c_3d_sub
            .as_slice()
            .iter()
            .zip(c_2d_contig.as_slice().iter())
            .enumerate()
        {
            assert!(
                (act - exp).abs() < 1e-5,
                "Batched GEMM batch {} index {} mismatch: {} vs {}",
                b,
                idx,
                act,
                exp
            );
        }
    }
}

#[test]
fn test_property_strided_coordinates_and_permutations() {
    let mut rng = StdRng::seed_from_u64(999);
    let shape = [3, 4, 5];
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| i as f32 + rng.gen_range(0.0..0.1))
        .collect();

    let tensor = RawTensor::from_slice(&data, &shape);

    // Verify logical indexing matches original linear assignment
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let expected = data[i * 20 + j * 5 + k];
                let actual = tensor.get(&[i, j, k]);
                assert_eq!(actual, expected);
            }
        }
    }

    // Permute to [5, 3, 4]
    let perm = tensor.permute(&[2, 0, 1]).unwrap();
    assert_eq!(perm.shape(), &[5, 3, 4]);

    for k in 0..5 {
        for i in 0..3 {
            for j in 0..4 {
                let expected = data[i * 20 + j * 5 + k];
                let actual = perm.get(&[k, i, j]);
                assert_eq!(
                    actual, expected,
                    "Permuted get mismatch at ({}, {}, {})",
                    k, i, j
                );
            }
        }
    }

    // Out-of-bounds rejection
    assert!(perm.try_get(&[5, 0, 0]).is_err());
    assert!(perm.try_get(&[0, 3, 0]).is_err());
    assert!(perm.try_get(&[0, 0, 4]).is_err());
    assert!(perm.try_get(&[0, 0]).is_err());
}

#[test]
fn test_property_broadcasting_arithmetic() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // [4, 1]
    let b_data = vec![10.0, 20.0, 30.0]; // [1, 3]

    let a = RawTensor::from_slice(&a_data, &[4, 1]);
    let b = RawTensor::from_slice(&b_data, &[1, 3]);

    let sum = a.add(&b).unwrap();
    assert_eq!(sum.shape(), &[4, 3]);

    for (r, &a_val) in a_data.iter().enumerate() {
        for (c, &b_val) in b_data.iter().enumerate() {
            let expected = a_val + b_val;
            assert_eq!(sum.get(&[r, c]), expected);
        }
    }

    let mul = a.mul(&b).unwrap();
    assert_eq!(mul.shape(), &[4, 3]);

    for (r, &a_val) in a_data.iter().enumerate() {
        for (c, &b_val) in b_data.iter().enumerate() {
            let expected = a_val * b_val;
            assert_eq!(mul.get(&[r, c]), expected);
        }
    }
}

#[test]
fn test_property_conv2d_vs_spatial_reference() {
    let mut rng = StdRng::seed_from_u64(777);
    let (h, w) = (7, 7);
    let (kh, kw) = (3, 3);

    let input_data: Vec<f32> = (0..h * w).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let kernel_data: Vec<f32> = (0..kh * kw).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let input = RawTensor::from_slice(&input_data, &[1, 1, h, w]);
    let kernel = RawTensor::from_slice(&kernel_data, &[1, 1, kh, kw]);

    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
    };
    let conv_out =
        neural_network_engine::tensor::conv::conv2d_forward(&input, &kernel, None, params).unwrap();
    let naive_out = naive_conv2d(&input_data, h, w, &kernel_data, kh, kw);

    let actual = conv_out.to_contiguous();
    for (idx, (&act, &exp)) in actual.as_slice().iter().zip(naive_out.iter()).enumerate() {
        assert!(
            (act - exp).abs() < 1e-4,
            "Conv2d mismatch at idx {}: {} vs {}",
            idx,
            act,
            exp
        );
    }
}
