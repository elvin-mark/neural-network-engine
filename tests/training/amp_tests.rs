use neural_network_engine::prelude::*;

#[test]
fn test_loss_scaler_basic_scaling_and_unscaling() {
    let scaler = LossScaler::new(1024.0);
    assert_eq!(scaler.current_scale(), 1024.0);

    let weight = Tensor::randn(&[4, 8], 0.0, 1.0, true);
    let x = Tensor::randn(&[2, 4], 0.0, 1.0, false);

    let out = x.matmul(&weight).unwrap();
    let loss = out.sum_all();

    // Scale loss
    let scaled_loss = scaler.scale(&loss).unwrap();
    scaled_loss.backward();

    // Gradient was computed with 1024.0x scale
    let unscaled_ok = scaler.unscale_grads(std::slice::from_ref(&weight));
    assert!(unscaled_ok);

    // Verify grad is finite
    let grad = weight.grad().unwrap();
    for &val in grad.to_contiguous().as_slice() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_loss_scaler_optimizer_step_integration() {
    let scaler = LossScaler::new(512.0);
    let linear = Linear::new(8, 4);
    let mut opt = SGD::new(linear.parameters(), 0.01);

    let x = Tensor::randn(&[2, 8], 0.0, 1.0, false);
    let out = linear.forward(&x).unwrap();
    let loss = out.sum_all();

    let scaled_loss = scaler.scale(&loss).unwrap();
    scaled_loss.backward();

    let stepped = scaler.step(&mut opt).unwrap();
    assert!(stepped, "Optimizer step should succeed with finite grads");
}

#[test]
fn test_loss_scaler_nan_backoff() {
    let scaler = LossScaler::new(1024.0);
    let param = Tensor::randn(&[2, 2], 0.0, 1.0, true);

    // Inject NaN into grad
    let nan_data = vec![f32::NAN, 1.0, 2.0, 3.0];
    param.set_grad(Some(RawTensor::from_vec(nan_data, vec![2, 2])));

    let mut opt = SGD::new(vec![param.clone()], 0.01);
    let stepped = scaler.step(&mut opt).unwrap();

    assert!(!stepped, "Step should be skipped when NaN grad is present");
    assert_eq!(
        scaler.current_scale(),
        512.0,
        "Scale should be halved after NaN"
    );
    assert!(param.grad().is_none(), "Grads should be zeroed after NaN");
}
