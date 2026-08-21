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

#[test]
fn test_vision_transformer_forward_backward() {
    let config = ViTConfig {
        image_size: 16,
        patch_size: 4,
        in_channels: 3,
        num_classes: 5,
        d_model: 16,
        num_layers: 2,
        num_heads: 2,
        mlp_dim: 32,
    };
    let vit = VisionTransformer::new(config);

    // Input batch: 2 images, 3 channels, 16x16
    let x = Tensor::randn(&[2, 3, 16, 16], 0.0, 1.0, true);
    let logits = vit.forward(&x).unwrap();

    assert_eq!(logits.shape(), &[2, 5]);

    // Backward pass
    let loss = logits.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &[2, 3, 16, 16]);
    assert!(vit.pos_embed.grad().is_some());
    assert_eq!(vit.pos_embed.grad().unwrap().shape(), &[1, 16, 16]);
    assert!(vit.head.weight.grad().is_some());
    assert_eq!(vit.head.weight.grad().unwrap().shape(), &[5, 16]);

    // Invalid input rank
    let rank3 = Tensor::randn(&[2, 3, 16], 0.0, 1.0, false);
    assert!(vit.forward(&rank3).is_err());

    // Invalid channels
    let wrong_ch = Tensor::randn(&[2, 1, 16, 16], 0.0, 1.0, false);
    assert!(vit.forward(&wrong_ch).is_err());
}
