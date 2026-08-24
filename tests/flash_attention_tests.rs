use neural_network_engine::prelude::*;

#[test]
fn test_flash_attention_parity_with_standard_attention_causal() {
    let b = 2;
    let h = 4;
    let t = 16;
    let d = 32;

    let q = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);
    let k = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);
    let v = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);

    // 1. FlashAttention-2 Output
    let flash_out = flash_attention_forward(&q, &k, &v, true, None, 8, 8).unwrap();
    assert_eq!(flash_out.shape(), &[b, h, t, d]);

    // 2. Reference Standard Attention Output
    // S = Q * K^T / sqrt(D)
    let q_tensor = Tensor::new(q, false);
    let k_tensor = Tensor::new(k, false);
    let v_tensor = Tensor::new(v, false);

    let k_t = k_tensor.transpose(2, 3).unwrap();
    let scores = q_tensor.matmul(&k_t).unwrap();
    let scale = 1.0 / (d as f32).sqrt();
    let mut scaled = scores.mul_scalar(scale).unwrap();

    // Apply causal mask
    let mask_data: Vec<f32> = (0..t)
        .flat_map(|r| (0..t).map(move |c| if c > r { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::new(RawTensor::from_vec(mask_data, vec![1, 1, t, t]), false);
    scaled = scaled.add(&mask).unwrap();

    let probs = scaled.softmax(3).unwrap();
    let std_out = probs.matmul(&v_tensor).unwrap();

    // 3. Compare numerical parity (cosine similarity)
    let flash_slice = flash_out.to_contiguous();
    let std_slice = std_out.data().to_contiguous();

    let mut dot = 0.0f32;
    let mut norm_flash = 0.0f32;
    let mut norm_std = 0.0f32;

    for (&a, &b_val) in flash_slice.as_slice().iter().zip(std_slice.as_slice()) {
        dot += a * b_val;
        norm_flash += a * a;
        norm_std += b_val * b_val;
    }

    let cos_sim = dot / (norm_flash.sqrt() * norm_std.sqrt());
    assert!(
        cos_sim > 0.9999,
        "FlashAttention parity failure: Cosine similarity was {}",
        cos_sim
    );
}

#[test]
fn test_flash_attention_non_causal() {
    let b = 1;
    let h = 2;
    let t = 32;
    let d = 16;

    let q = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);
    let k = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);
    let v = RawTensor::randn(&[b, h, t, d], 0.0, 1.0);

    let out = flash_attention_forward(&q, &k, &v, false, None, 16, 16).unwrap();
    assert_eq!(out.shape(), &[b, h, t, d]);
}

#[test]
fn test_flash_attention_layer_forward() {
    let layer = FlashAttention::new(64, 4, true);
    let x = Tensor::randn(&[2, 16, 64], 0.0, 1.0, true);

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 16, 64]);
}
