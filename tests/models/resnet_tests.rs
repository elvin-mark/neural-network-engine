use neural_network_engine::prelude::*;

#[test]
fn test_batchnorm2d_forward_backward() {
    let mut bn = BatchNorm2d::new(4);
    bn.train();

    let x = Tensor::randn(&[2, 4, 8, 8], 5.0, 2.0, true);
    let out = bn.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 4, 8, 8]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());

    // Switch to eval mode
    bn.eval();
    let eval_x = Tensor::randn(&[2, 4, 8, 8], 0.0, 1.0, false);
    let eval_out = bn.forward(&eval_x).unwrap();
    assert_eq!(eval_out.shape(), &[2, 4, 8, 8]);
}

#[test]
fn test_residual_block_forward_backward() {
    // 1. Without downsample shortcut (stride 1, in_channels == out_channels)
    let block = ResidualBlock::new(16, 16, 1);
    let x = Tensor::randn(&[2, 16, 8, 8], 0.0, 1.0, true);

    let out = block.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 16, 8, 8]);

    let loss = out.sum_all();
    loss.backward();
    assert!(x.grad().is_some());
    assert!(block.conv1.weight.grad().is_some());
    assert!(block.conv2.weight.grad().is_some());

    // 2. With downsample shortcut (stride 2, in_channels != out_channels)
    let ds_block = ResidualBlock::new(16, 32, 2);
    let x2 = Tensor::randn(&[2, 16, 8, 8], 0.0, 1.0, true);

    let out2 = ds_block.forward(&x2).unwrap();
    assert_eq!(out2.shape(), &[2, 32, 4, 4]);

    let loss2 = out2.sum_all();
    loss2.backward();
    assert!(x2.grad().is_some());
    assert!(ds_block.downsample.is_some());
}

#[test]
fn test_bottleneck_block_forward_backward() {
    // Expansion = 4, so out_channels=16 -> final channels = 64
    let block = BottleneckBlock::new(64, 16, 1);
    let x = Tensor::randn(&[2, 64, 8, 8], 0.0, 1.0, true);

    let out = block.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 64, 8, 8]);

    let loss = out.sum_all();
    loss.backward();
    assert!(x.grad().is_some());

    // With stride 2 downsampling
    let ds_block = BottleneckBlock::new(64, 32, 2);
    let x2 = Tensor::randn(&[2, 64, 8, 8], 0.0, 1.0, true);
    let out2 = ds_block.forward(&x2).unwrap();
    assert_eq!(out2.shape(), &[2, 128, 4, 4]); // 32 * 4 = 128
}

#[test]
fn test_cifar_resnet18_forward_backward() {
    let resnet = ResNet::cifar_resnet18(3, 10);
    let x = Tensor::randn(&[2, 3, 32, 32], 0.0, 1.0, true);

    let logits = resnet.forward(&x).unwrap();
    assert_eq!(logits.shape(), &[2, 10]);

    let loss = logits.sum_all();
    loss.backward();
    assert!(x.grad().is_some());
    assert!(resnet.fc.weight.grad().is_some());
}

#[test]
fn test_resnet18_imagenet_stem() {
    let resnet = ResNet::resnet18(3, 1000);
    let x = Tensor::randn(&[1, 3, 64, 64], 0.0, 1.0, false);

    let logits = resnet.forward(&x).unwrap();
    assert_eq!(logits.shape(), &[1, 1000]);
}
