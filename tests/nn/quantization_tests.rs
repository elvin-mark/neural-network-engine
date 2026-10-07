use neural_network_engine::prelude::*;

#[test]
fn test_int8_tensor_roundtrip() {
    let raw = RawTensor::randn(&[32, 64], 0.0, 1.0);
    let q_tensor = Int8Tensor::from_raw(&raw);

    assert_eq!(q_tensor.shape, vec![32, 64]);
    assert_eq!(q_tensor.data.len(), 32 * 64);
    assert!(q_tensor.scale > 0.0);

    let dequant = q_tensor.dequantize();
    assert_eq!(dequant.shape(), raw.shape());

    // Compute reconstruction RMSE
    let orig_slice = raw.to_contiguous();
    let deq_slice = dequant.to_contiguous();
    let mut mse = 0.0f32;
    for (&orig, &deq) in orig_slice.as_slice().iter().zip(deq_slice.as_slice()) {
        let diff = orig - deq;
        mse += diff * diff;
    }
    let rmse = (mse / (32.0 * 64.0)).sqrt();
    assert!(
        rmse < 0.05,
        "INT8 quantization error too high: RMSE = {}",
        rmse
    );
}

#[test]
fn test_qlinear_from_linear_and_forward() {
    let linear = Linear::new(64, 128);
    let qlinear = QLinear::from_linear(&linear);

    assert_eq!(qlinear.in_features, 64);
    assert_eq!(qlinear.out_features, 128);
    assert_eq!(qlinear.qweight.len(), 64 * 128);
    assert_eq!(qlinear.scales.len(), 128);

    // 1. 2D Input [Batch=8, Features=64]
    let x_2d = Tensor::randn(&[8, 64], 0.0, 1.0, false);
    let out_fp32 = linear.forward(&x_2d).unwrap();
    let out_int8 = qlinear.forward(&x_2d).unwrap();

    assert_eq!(out_fp32.shape(), out_int8.shape());

    // Cosine similarity / relative accuracy check
    let fp32_slice = out_fp32.data().to_contiguous();
    let int8_slice = out_int8.data().to_contiguous();

    let mut dot = 0.0f32;
    let mut norm_fp32 = 0.0f32;
    let mut norm_int8 = 0.0f32;
    for (&a, &b) in fp32_slice.as_slice().iter().zip(int8_slice.as_slice()) {
        dot += a * b;
        norm_fp32 += a * a;
        norm_int8 += b * b;
    }
    let cos_sim = dot / (norm_fp32.sqrt() * norm_int8.sqrt());
    assert!(
        cos_sim > 0.995,
        "QLinear 2D cosine similarity too low: {}",
        cos_sim
    );

    // 2. 3D Input [Batch=4, SeqLen=10, Features=64]
    let x_3d = Tensor::randn(&[4, 10, 64], 0.0, 1.0, false);
    let out_3d_fp32 = linear.forward(&x_3d).unwrap();
    let out_3d_int8 = qlinear.forward(&x_3d).unwrap();

    assert_eq!(out_3d_fp32.shape(), out_3d_int8.shape());
    assert_eq!(out_3d_int8.shape(), &[4, 10, 128]);
}

#[test]
fn test_qlinear_memory_compression() {
    let linear = Linear::new(256, 512);
    let qlinear = QLinear::from_linear(&linear);

    let fp32_bytes = 256 * 512 * std::mem::size_of::<f32>() + 512 * std::mem::size_of::<f32>();
    let int8_bytes = qlinear.memory_bytes();

    let compression_ratio = fp32_bytes as f32 / int8_bytes as f32;
    assert!(
        compression_ratio > 3.8,
        "Compression ratio {} is less than expected ~4x",
        compression_ratio
    );
}

#[test]
fn test_qlinear_to_linear_roundtrip() {
    let linear = Linear::new(32, 64);
    let qlinear = QLinear::from_linear(&linear);
    let recovered_linear = qlinear.to_linear();

    assert_eq!(recovered_linear.in_features, 32);
    assert_eq!(recovered_linear.out_features, 64);

    let x = Tensor::randn(&[2, 32], 0.0, 1.0, false);
    let out_q = qlinear.forward(&x).unwrap();
    let out_rec = recovered_linear.forward(&x).unwrap();

    let q_slice = out_q.data().to_contiguous();
    let rec_slice = out_rec.data().to_contiguous();

    for (a, b) in q_slice.as_slice().iter().zip(rec_slice.as_slice()) {
        assert!(
            (a - b).abs() < 1e-4,
            "Recovered linear mismatch: {} vs {}",
            a,
            b
        );
    }
}
