#![cfg(feature = "gpu")]

use neural_network_engine::prelude::*;

fn assert_tensor_approx_eq(a: &RawTensor, b: &RawTensor, tol: f32) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "Shapes mismatch: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let a_slice = a.to_contiguous();
    let b_slice = b.to_contiguous();
    for (idx, (&x, &y)) in a_slice
        .as_slice()
        .iter()
        .zip(b_slice.as_slice().iter())
        .enumerate()
    {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "Values mismatch at index {}: CPU {} vs GPU {} (diff: {} > tol: {})",
            idx,
            x,
            y,
            diff,
            tol
        );
    }
}

#[test]
fn test_gpu_context_and_roundtrip() {
    let ctx = GpuContext::new().expect("Failed to initialize GPU context");
    println!("Testing on GPU Device: {}", ctx.adapter_info.name);

    let cpu_tensor = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gpu_tensor = cpu_tensor
        .to_gpu(&ctx)
        .expect("Failed to upload tensor to GPU");

    assert_eq!(gpu_tensor.shape(), &[2, 3]);
    assert_eq!(gpu_tensor.numel(), 6);

    let downloaded = gpu_tensor
        .to_cpu()
        .expect("Failed to download tensor to CPU");
    assert_tensor_approx_eq(&cpu_tensor, &downloaded, 1e-5);
}

#[test]
fn test_gpu_matmul_numerical_parity() {
    let ctx = GpuContext::new().expect("Failed to initialize GPU context");

    let sizes = [(16, 32, 24), (64, 128, 48), (128, 64, 128), (37, 53, 29)]; // Includes non-multiples of 16

    for (m, k, n) in sizes {
        let a_cpu = RawTensor::randn(&[m, k], 0.0, 1.0);
        let b_cpu = RawTensor::randn(&[k, n], 0.0, 1.0);
        let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

        let a_gpu = a_cpu.to_gpu(&ctx).unwrap();
        let b_gpu = b_cpu.to_gpu(&ctx).unwrap();
        let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

        let c_gpu_downloaded = c_gpu.to_cpu().unwrap();
        assert_tensor_approx_eq(&c_cpu, &c_gpu_downloaded, 1e-3);
    }
}

#[test]
fn test_gpu_elementwise_and_activations() {
    let ctx = GpuContext::new().expect("Failed to initialize GPU context");

    let a_cpu = RawTensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]);
    let b_cpu = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let a_gpu = a_cpu.to_gpu(&ctx).unwrap();
    let b_gpu = b_cpu.to_gpu(&ctx).unwrap();

    // 1. Add
    let add_cpu = a_cpu.add(&b_cpu).unwrap();
    let add_gpu = a_gpu.add(&b_gpu).unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&add_cpu, &add_gpu, 1e-5);

    // 2. Mul
    let mul_cpu = a_cpu.mul(&b_cpu).unwrap();
    let mul_gpu = a_gpu.mul(&b_gpu).unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&mul_cpu, &mul_gpu, 1e-5);

    // 3. ReLU
    let relu_cpu = a_cpu.relu().unwrap();
    let relu_gpu = a_gpu.relu().unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&relu_cpu, &relu_gpu, 1e-5);

    // 4. GELU
    let gelu_cpu = a_cpu.gelu().unwrap();
    let gelu_gpu = a_gpu.gelu().unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&gelu_cpu, &gelu_gpu, 1e-4);

    // 5. SiLU
    let silu_cpu = a_cpu.silu().unwrap();
    let silu_gpu = a_gpu.silu().unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&silu_cpu, &silu_gpu, 1e-4);

    // 6. Scale
    let scale_cpu = a_cpu.mul_scalar(3.5).unwrap();
    let scale_gpu = a_gpu.scale(3.5).unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&scale_cpu, &scale_gpu, 1e-5);
}

#[test]
fn test_gpu_softmax_and_layernorm() {
    let ctx = GpuContext::new().expect("Failed to initialize GPU context");

    let x_cpu = RawTensor::randn(&[8, 32], 0.0, 1.0);
    let x_gpu = x_cpu.to_gpu(&ctx).unwrap();

    // 1. Softmax
    let sm_cpu = x_cpu.softmax(1).unwrap();
    let sm_gpu = x_gpu.softmax().unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&sm_cpu, &sm_gpu, 1e-4);

    // 2. LayerNorm
    let ln_cpu = LayerNorm::new(32);
    let ln_gpu = ln_cpu.to_gpu(&ctx).unwrap();

    let out_cpu = ln_cpu.forward(&Tensor::new(x_cpu.clone(), false)).unwrap();
    let out_gpu = ln_gpu.forward(&x_gpu).unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&out_cpu.data(), &out_gpu, 1e-4);

    // 3. RMSNorm
    let rms_cpu = RMSNorm::new(32);
    let rms_gpu = rms_cpu.to_gpu(&ctx).unwrap();

    let out_rms_cpu = rms_cpu.forward(&Tensor::new(x_cpu, false)).unwrap();
    let out_rms_gpu = rms_gpu.forward(&x_gpu).unwrap().to_cpu().unwrap();
    assert_tensor_approx_eq(&out_rms_cpu.data(), &out_rms_gpu, 1e-4);
}

#[test]
fn test_gpu_linear_layer() {
    let ctx = GpuContext::new().expect("Failed to initialize GPU context");

    let linear_cpu = Linear::new(64, 128);
    let linear_gpu = linear_cpu.to_gpu(&ctx).unwrap();

    let x_cpu = RawTensor::randn(&[16, 64], 0.0, 1.0);
    let x_gpu = x_cpu.to_gpu(&ctx).unwrap();

    let y_cpu = linear_cpu.forward(&Tensor::new(x_cpu, false)).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap().to_cpu().unwrap();

    assert_tensor_approx_eq(&y_cpu.data(), &y_gpu, 1e-3);
}
