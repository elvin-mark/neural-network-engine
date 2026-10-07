use neural_network_engine::prelude::*;

#[test]
fn test_random_horizontal_flip_deterministic() {
    let flip_always = RandomHorizontalFlip::new(1.0);
    let flip_never = RandomHorizontalFlip::new(0.0);

    // 1 channel, 2x2 image: [[1.0, 2.0], [3.0, 4.0]]
    let img = RawTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);

    let flipped = flip_always.apply_raw(&img).unwrap();
    let unflipped = flip_never.apply_raw(&img).unwrap();

    assert_eq!(
        flipped.to_contiguous().as_slice(),
        &[2.0, 1.0, 4.0, 3.0] // flipped columns
    );
    assert_eq!(unflipped.to_contiguous().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_random_vertical_flip_deterministic() {
    let flip_always = RandomVerticalFlip::new(1.0);

    // 1 channel, 2x2 image: [[1.0, 2.0], [3.0, 4.0]]
    let img = RawTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let flipped = flip_always.apply_raw(&img).unwrap();

    assert_eq!(
        flipped.to_contiguous().as_slice(),
        &[3.0, 4.0, 1.0, 2.0] // flipped rows
    );
}

#[test]
fn test_random_crop_and_padding() {
    let crop = RandomCrop::new(4, 4, 2);
    let img = RawTensor::ones(&[3, 4, 4]);

    let cropped = crop.apply_raw(&img).unwrap();
    assert_eq!(cropped.shape(), &[3, 4, 4]);
}

#[test]
fn test_normalize_cifar10() {
    let norm = Normalize::cifar10();
    // 3 channels of 2x2 with values equal to means
    let img = RawTensor::from_vec(
        vec![
            0.4914, 0.4914, 0.4914, 0.4914, // Channel 0
            0.4822, 0.4822, 0.4822, 0.4822, // Channel 1
            0.4465, 0.4465, 0.4465, 0.4465, // Channel 2
        ],
        vec![3, 2, 2],
    );

    let normalized = norm.apply_raw(&img).unwrap();
    let slice = normalized.to_contiguous();

    // Since input == mean, normalized output must be ~0.0
    for &val in slice.as_slice() {
        assert!(val.abs() < 1e-4, "Expected ~0.0, got {}", val);
    }
}

#[test]
fn test_compose_pipeline() {
    let pipeline = Compose::new(vec![
        Box::new(RandomHorizontalFlip::new(1.0)),
        Box::new(RandomCrop::new(4, 4, 1)),
        Box::new(Normalize::new(vec![0.5], vec![0.5])),
    ]);

    let x = Tensor::randn(&[1, 1, 4, 4], 0.5, 0.1, false);
    let transformed = pipeline.apply(&x).unwrap();

    assert_eq!(transformed.shape(), &[1, 1, 4, 4]);
}
