use neural_network_engine::prelude::*;

#[test]
fn test_linear_layer() {
    let linear = Linear::new(4, 2);
    let x = Tensor::randn(&[3, 4], 0.0, 1.0, true);

    let out = linear.forward(&x).unwrap();
    assert_eq!(out.shape(), &[3, 2]);

    let loss = out.sum_all();
    loss.backward();

    assert!(linear.weight.grad().is_some());
    assert_eq!(linear.weight.grad().unwrap().shape(), &[2, 4]);
    assert!(linear.bias.as_ref().unwrap().grad().is_some());
    assert_eq!(linear.bias.as_ref().unwrap().grad().unwrap().shape(), &[2]);
}

#[test]
fn test_conv2d_layer() {
    let conv = Conv2d::new(2, 4, (3, 3));
    let x = Tensor::randn(&[2, 2, 8, 8], 0.0, 1.0, true);

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 6, 6]);

    let loss = out.sum_all();
    loss.backward();

    assert!(conv.weight.grad().is_some());
    assert_eq!(conv.weight.grad().unwrap().shape(), &[4, 2, 3, 3]);
}

#[test]
fn test_maxpool2d_layer() {
    let pool = MaxPool2d::square(2);
    let x = Tensor::randn(&[2, 3, 8, 8], 0.0, 1.0, true);

    let out = pool.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3, 4, 4]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &[2, 3, 8, 8]);
}

#[test]
fn test_sequential_container() {
    let seq = Sequential::new()
        .add(Linear::new(10, 20))
        .add(ReLU)
        .add(Linear::new(20, 5));

    let x = Tensor::randn(&[4, 10], 0.0, 1.0, true);
    let out = seq.forward(&x).unwrap();
    assert_eq!(out.shape(), &[4, 5]);

    let params = seq.parameters();
    assert_eq!(params.len(), 4); // (weight + bias) * 2
}

#[test]
fn test_optimizers_step() {
    let x = Tensor::new(RawTensor::from_slice(&[2.0, -3.0], &[2]), true);
    let mut sgd = SGD::new(vec![x.clone()], 0.1).with_momentum(0.9);

    // Compute loss = sum(x^2)
    let x2 = x.mul(&x).unwrap();
    let loss = x2.sum_all();
    loss.backward();

    sgd.step().unwrap();

    // Initial x: [2, -3]
    // Grad: [4, -6]
    // Step: x = x - 0.1 * [4, -6] = [1.6, -2.4]
    let new_x = x.data();
    assert!((new_x.get(&[0]) - 1.6).abs() < 1e-4);
    assert!((new_x.get(&[1]) - (-2.4)).abs() < 1e-4);
}
