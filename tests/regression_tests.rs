//! Regression tests for BatchNorm running stats, repeated backward calls, SafeTensors validation,
//! input validation in permute/loss, and data loader edge cases.

use neural_network_engine::prelude::*;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

#[test]
fn test_batchnorm1d_running_statistics_update() {
    let mut bn = BatchNorm1d::new(2);
    assert_eq!(bn.running_mean().as_slice(), &[0.0, 0.0]);
    assert_eq!(bn.running_var().as_slice(), &[1.0, 1.0]);

    // Batch with mean [4.0, 6.0] and variance [9.0, 16.0]
    // Dim 0: values 1.0 and 7.0 -> mean 4.0, diffs -3, +3 -> var = 9.0
    // Dim 1: values 2.0 and 10.0 -> mean 6.0, diffs -4, +4 -> var = 16.0
    let input = Tensor::new(
        RawTensor::from_slice(&[1.0, 2.0, 7.0, 10.0], &[2, 2]),
        false,
    );

    // Forward pass in training mode
    let _out = bn.forward(&input).unwrap();

    let rm = bn.running_mean();
    let rv = bn.running_var();

    // running_mean should be (1 - 0.1)*0 + 0.1*4.0 = 0.4 for dim 0, and 0.6 for dim 1
    assert!((rm.as_slice()[0] - 0.4).abs() < 1e-4);
    assert!((rm.as_slice()[1] - 0.6).abs() < 1e-4);

    // running_var should be (1 - 0.1)*1.0 + 0.1*9.0 = 1.8 for dim 0, and (1-0.1)*1.0 + 0.1*16.0 = 2.5 for dim 1
    assert!((rv.as_slice()[0] - 1.8).abs() < 1e-4);
    assert!((rv.as_slice()[1] - 2.5).abs() < 1e-4);

    // Switch to eval mode
    bn.eval();
    let eval_input = Tensor::new(RawTensor::from_slice(&[0.4, 0.6], &[1, 2]), false);
    let eval_out = bn.forward(&eval_input).unwrap();
    // In eval mode, input equal to running_mean should normalize to 0.0 (+ beta 0.0 = 0.0)
    assert!(eval_out.data().as_slice()[0].abs() < 1e-4);
    assert!(eval_out.data().as_slice()[1].abs() < 1e-4);
}

#[test]
fn test_repeated_backward_accumulation() {
    // 1. Single leaf node with repeated backward calls
    let x = Tensor::new(RawTensor::from_slice(&[2.0, 3.0], &[2]), true);
    let loss1 = x.mul_scalar(2.0).unwrap().sum_all();

    loss1.backward();
    assert_eq!(x.grad().unwrap().as_slice(), &[2.0, 2.0]);

    // Second backward call should accumulate linearly (+2.0 -> 4.0), not exponentially
    loss1.backward();
    assert_eq!(x.grad().unwrap().as_slice(), &[4.0, 4.0]);

    // 2. Multi-layer graph with intermediate nonlinear activations: y = x^2
    let a = Tensor::new(RawTensor::from_slice(&[3.0], &[1]), true);
    let b = a.mul(&a).unwrap(); // b = 9.0, db/da = 2*a = 6.0
    let loss2 = b.sum_all();

    loss2.backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[6.0]);

    loss2.backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[12.0]);
}

#[test]
fn test_permute_validation() {
    let t = RawTensor::zeros(&[2, 3]);

    // Valid permutation
    assert!(t.permute(&[1, 0]).is_ok());
    assert_eq!(t.permute(&[1, 0]).unwrap().shape(), &[3, 2]);

    // Invalid permutations
    assert!(t.permute(&[0, 0]).is_err());
    assert!(t.permute(&[1, 1]).is_err());
    assert!(t.permute(&[0, 2]).is_err());
    assert!(t.permute(&[0]).is_err());
    assert!(t.permute(&[0, 1, 2]).is_err());
}

#[test]
fn test_crossentropy_loss_validation() {
    let logits_2d = Tensor::zeros(&[3, 4], true);

    // Valid forward
    assert!(CrossEntropyLoss::forward_with_indices(&logits_2d, &[0, 1, 3]).is_ok());

    // Batch size mismatch (expected 3, provided 2)
    assert!(CrossEntropyLoss::forward_with_indices(&logits_2d, &[0, 1]).is_err());

    // Target class out of bounds (num_classes is 4, target is 5)
    assert!(CrossEntropyLoss::forward_with_indices(&logits_2d, &[0, 5, 2]).is_err());

    // Non-2D logits
    let logits_3d = Tensor::zeros(&[2, 3, 4], true);
    assert!(CrossEntropyLoss::forward_with_indices(&logits_3d, &[0, 1]).is_err());
}

#[test]
fn test_safetensors_type_safety_and_validation() {
    // 1. Valid roundtrip
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );
    let temp_path = "target/test_valid.safetensors";
    save_safetensors(&tensors, temp_path).unwrap();

    let loaded = load_safetensors(temp_path).unwrap();
    assert_eq!(
        loaded.get("weight").unwrap().as_slice(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    // 2. Reject non-F32 dtype (e.g. I32)
    let invalid_path = "target/test_invalid_dtype.safetensors";
    let fake_bytes = vec![0u8; 16];
    let mut views = HashMap::new();
    views.insert(
        "invalid".to_string(),
        TensorView::new(Dtype::I32, vec![4], &fake_bytes).unwrap(),
    );
    let serialized = safetensors::serialize(&views, &None).unwrap();
    let mut file = File::create(invalid_path).unwrap();
    file.write_all(&serialized).unwrap();

    let result = load_safetensors(invalid_path);
    assert!(
        result.is_err(),
        "Expected load_safetensors to reject non-F32 dtype"
    );
}

#[test]
fn test_data_utils_hardening_and_contiguity() {
    // Non-contiguous features tensor
    let orig = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let non_contig = orig.permute(&[1, 0]).unwrap(); // [2, 3] non-contiguous view
    assert!(!non_contig.is_contiguous());

    // standardize should handle non-contiguous input gracefully
    let (std_tensor, mean, std) = standardize(&non_contig);
    assert_eq!(std_tensor.shape(), &[2, 3]);
    assert_eq!(mean.len(), 3);
    assert_eq!(std.len(), 3);

    // train_test_split should handle non-contiguous input gracefully
    let labels = vec![0, 1];
    let (train_f, train_l, test_f, test_l) = train_test_split(&non_contig, &labels, 0.5, false);
    assert_eq!(train_f.shape(), &[1, 3]);
    assert_eq!(test_f.shape(), &[1, 3]);
    assert_eq!(train_l.len(), 1);
    assert_eq!(test_l.len(), 1);

    // TensorDataset should convert non-contiguous tensors to contiguous buffers
    let dataset = TensorDataset::new(non_contig, RawTensor::zeros(&[2, 1])).unwrap();
    assert!(dataset.features.is_contiguous());
}

#[test]
fn test_csv_dataset_loaders() {
    // 1. Iris CSV parser test
    let iris_csv = "5.1,3.5,1.4,0.2,Iris-setosa\n6.7,3.0,5.0,1.7,Iris-versicolor\n6.3,3.3,6.0,2.5,Iris-virginica\n";
    let iris_path = "target/test_iris.csv";
    let mut f1 = File::create(iris_path).unwrap();
    f1.write_all(iris_csv.as_bytes()).unwrap();

    let (x_iris, y_iris) = load_iris_from_csv(iris_path).unwrap();
    assert_eq!(x_iris.shape(), &[3, 4]);
    assert_eq!(y_iris, vec![0, 1, 2]);

    // 2. Digits CSV parser test (64 zeros + label 7)
    let mut digits_line = vec!["0"; 64];
    digits_line[10] = "16"; // 16 / 16 = 1.0
    let mut line_str = digits_line.join(",");
    line_str.push_str(",7\n");

    let digits_path = "target/test_digits.csv";
    let mut f2 = File::create(digits_path).unwrap();
    f2.write_all(line_str.as_bytes()).unwrap();

    let (x_digits, y_digits) = load_digits_from_csv(digits_path, None).unwrap();
    assert_eq!(x_digits.shape(), &[1, 1, 8, 8]);
    assert_eq!(y_digits, vec![7]);
    assert!((x_digits.as_slice()[10] - 1.0).abs() < 1e-4);
}

#[test]
#[should_panic(expected = "DataLoader batch_size must be greater than 0")]
fn test_dataloader_zero_batch_size_panics() {
    let features = RawTensor::zeros(&[10, 2]);
    let targets = RawTensor::zeros(&[10, 1]);
    let dataset = TensorDataset::new(features, targets).unwrap();
    let _loader = DataLoader::new(&dataset, 0, false);
}
