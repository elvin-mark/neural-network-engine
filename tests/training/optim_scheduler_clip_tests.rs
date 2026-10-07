use neural_network_engine::prelude::*;

#[test]
fn test_gradient_clipping_norm() {
    let p1 = Tensor::new(RawTensor::from_slice(&[3.0, 4.0], &[2]), true);
    p1.set_grad(Some(RawTensor::from_slice(&[3.0, 4.0], &[2]))); // Norm is sqrt(9 + 16) = 5.0

    let p2 = Tensor::new(RawTensor::from_slice(&[0.0, 0.0], &[2]), true);
    p2.set_grad(Some(RawTensor::from_slice(&[0.0, 0.0], &[2])));

    let params = vec![p1.clone(), p2.clone()];

    // Clip norm to 2.5 (should scale gradients by 2.5 / 5.0 = 0.5)
    let total_norm = clip_grad_norm(&params, 2.5);
    assert!((total_norm - 5.0).abs() < 1e-4);

    let grad1 = p1.grad().unwrap();
    assert!((grad1.as_slice()[0] - 1.5).abs() < 1e-4);
    assert!((grad1.as_slice()[1] - 2.0).abs() < 1e-4);
}

#[test]
fn test_gradient_clipping_value() {
    let p = Tensor::new(RawTensor::from_slice(&[1.0, 2.0, 3.0], &[3]), true);
    p.set_grad(Some(RawTensor::from_slice(&[-5.0, 0.5, 10.0], &[3])));

    let params = vec![p.clone()];
    clip_grad_value(&params, 2.0);

    let grad = p.grad().unwrap();
    assert_eq!(grad.as_slice(), &[-2.0, 0.5, 2.0]);
}

#[test]
fn test_step_lr_scheduler() {
    let p = Tensor::new(RawTensor::zeros(&[2, 2]), true);
    let mut opt = SGD::new(vec![p], 0.1);
    let mut scheduler = StepLR::new(0.1, 2, 0.5);

    assert_eq!(opt.get_lr(), 0.1);

    // Epoch 1 (step_size is 2, so epoch 1 has no decay: 0.1)
    let lr1 = scheduler.step(&mut opt);
    assert!((lr1 - 0.1).abs() < 1e-5);
    assert_eq!(opt.get_lr(), lr1);

    // Epoch 2 (decay by 0.5 -> 0.05)
    let lr2 = scheduler.step(&mut opt);
    assert!((lr2 - 0.05).abs() < 1e-5);
    assert_eq!(opt.get_lr(), lr2);

    // Epoch 3 (no decay -> 0.05)
    let lr3 = scheduler.step(&mut opt);
    assert!((lr3 - 0.05).abs() < 1e-5);

    // Epoch 4 (decay by 0.5 -> 0.025)
    let lr4 = scheduler.step(&mut opt);
    assert!((lr4 - 0.025).abs() < 1e-5);
}

#[test]
fn test_multistep_lr_scheduler() {
    let p = Tensor::new(RawTensor::zeros(&[2, 2]), true);
    let mut opt = Adam::new(vec![p], 0.1);
    let mut scheduler = MultiStepLR::new(0.1, vec![2, 4], 0.1);

    assert_eq!(scheduler.step(&mut opt), 0.1); // Epoch 1
    assert!((scheduler.step(&mut opt) - 0.01).abs() < 1e-5); // Epoch 2 (milestone)
    assert!((scheduler.step(&mut opt) - 0.01).abs() < 1e-5); // Epoch 3
    assert!((scheduler.step(&mut opt) - 0.001).abs() < 1e-5); // Epoch 4 (milestone)
}

#[test]
fn test_exponential_lr_scheduler() {
    let p = Tensor::new(RawTensor::zeros(&[2, 2]), true);
    let mut opt = SGD::new(vec![p], 0.1);
    let mut scheduler = ExponentialLR::new(0.1, 0.9);

    assert!((scheduler.step(&mut opt) - 0.09).abs() < 1e-5); // Epoch 1
    assert!((scheduler.step(&mut opt) - 0.081).abs() < 1e-5); // Epoch 2
}

#[test]
fn test_cosine_annealing_lr() {
    let p = Tensor::new(RawTensor::zeros(&[2, 2]), true);
    let mut opt = Adam::new(vec![p], 0.1);
    let mut scheduler = CosineAnnealingLR::new(0.1, 10, 0.0);

    // After 5 steps (halfway), lr should be exactly half (0.05)
    for _ in 0..5 {
        scheduler.step(&mut opt);
    }
    assert!((opt.get_lr() - 0.05).abs() < 1e-4);

    // After 10 steps (full T_max), lr should be 0.0
    for _ in 5..10 {
        scheduler.step(&mut opt);
    }
    assert!(opt.get_lr().abs() < 1e-4);
}

#[test]
fn test_linear_warmup_cosine_lr() {
    let p = Tensor::new(RawTensor::zeros(&[2, 2]), true);
    let mut opt = Adam::new(vec![p], 0.0);
    let mut scheduler = LinearWarmupCosineLR::new(0.1, 5, 20, 0.001, 0.0);

    // Warmup step 5 (reaches base_lr 0.1)
    for _ in 0..5 {
        scheduler.step(&mut opt);
    }
    assert!((opt.get_lr() - 0.1).abs() < 1e-5);

    // End of schedule (step 20, reaches min_lr 0.001)
    for _ in 5..20 {
        scheduler.step(&mut opt);
    }
    assert!((opt.get_lr() - 0.001).abs() < 1e-5);
}
