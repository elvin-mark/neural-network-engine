use neural_network_engine::prelude::*;

#[test]
fn test_tensor_creation_and_indexing() {
    let t = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.get(&[0, 1]), 2.0);
    assert_eq!(t.get(&[1, 2]), 6.0);
}

#[test]
fn test_tensor_broadcasting_arithmetic() {
    // [2, 3] + [1, 3] -> [2, 3]
    let a = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = RawTensor::from_slice(&[10.0, 20.0, 30.0], &[1, 3]);
    let c = a.add(&b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.get(&[0, 0]), 11.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[0, 2]), 33.0);
    assert_eq!(c.get(&[1, 0]), 14.0);
    assert_eq!(c.get(&[1, 1]), 25.0);
    assert_eq!(c.get(&[1, 2]), 36.0);
}

#[test]
fn test_gemm_2d() {
    // [2, 3] * [3, 2] -> [2, 2]
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7, 8],
    //      [9, 1],
    //      [2, 3]]
    // C[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    let a = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = RawTensor::from_slice(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0], &[3, 2]);
    let c = neural_network_engine::tensor::matmul::matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 31.0);
    assert_eq!(c.get(&[0, 1]), 19.0);
    assert_eq!(c.get(&[1, 0]), 85.0);
    assert_eq!(c.get(&[1, 1]), 55.0);
}

#[test]
fn test_reductions() {
    let t = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.sum_all(), 21.0);
    assert_eq!(t.mean_all(), 3.5);

    let sum_dim0 = t.sum(0, false).unwrap();
    assert_eq!(sum_dim0.shape(), &[3]);
    assert_eq!(sum_dim0.get(&[0]), 5.0);
    assert_eq!(sum_dim0.get(&[1]), 7.0);
    assert_eq!(sum_dim0.get(&[2]), 9.0);

    let sum_dim1 = t.sum(1, false).unwrap();
    assert_eq!(sum_dim1.shape(), &[2]);
    assert_eq!(sum_dim1.get(&[0]), 6.0);
    assert_eq!(sum_dim1.get(&[1]), 15.0);
}

#[test]
fn test_slicing_and_concatenation() {
    let a = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = RawTensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let cat0 = RawTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(cat0.shape(), &[4, 2]);

    let cat1 = RawTensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(cat1.shape(), &[2, 4]);

    let slice = cat0.slice(0, 1, 3).unwrap();
    assert_eq!(slice.shape(), &[2, 2]);
    let contig = slice.to_contiguous();
    assert_eq!(contig.as_slice(), &[3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_gemv_a_bt_m1_parity() {
    use neural_network_engine::tensor::matmul::{matmul, matmul_transposed_b};

    // Test multiple non-trivial shapes (powers of two, odd numbers, primes)
    let test_cases = [(1, 7, 13), (1, 32, 64), (1, 65, 127), (1, 384, 512)];

    for (m, k, n) in test_cases {
        let a_vals: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.17).sin()).collect();
        let b_vals: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.31).cos()).collect();

        let a = RawTensor::from_slice(&a_vals, &[m, k]);
        let b = RawTensor::from_slice(&b_vals, &[n, k]);

        // 1. Reference using standard matmul with explicitly transposed B
        let b_t = b.transpose(0, 1).unwrap().to_contiguous();
        let ref_c = matmul(&a, &b_t).unwrap();

        // 2. Transposed GEMV fast-path C = A * B^T
        let fast_c = matmul_transposed_b(&a, &b).unwrap();

        assert_eq!(ref_c.shape(), fast_c.shape());
        assert_eq!(fast_c.shape(), &[m, n]);

        let ref_slice = ref_c.as_slice();
        let fast_slice = fast_c.as_slice();

        for i in 0..ref_slice.len() {
            let diff = (ref_slice[i] - fast_slice[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {} for shape [{}, {}, {}]: ref {} vs fast {}",
                i,
                m,
                k,
                n,
                ref_slice[i],
                fast_slice[i]
            );
        }
    }
}
