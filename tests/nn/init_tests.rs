use neural_network_engine::prelude::*;

#[test]
fn test_fan_in_and_fan_out_calculation() {
    // 1D
    assert_eq!(calculate_fan_in_and_fan_out(&[128]), (128, 128));

    // 2D Linear [out_features, in_features]
    assert_eq!(calculate_fan_in_and_fan_out(&[64, 128]), (128, 64));

    // 4D Conv2D [out_channels, in_channels, kH, kW]
    assert_eq!(
        calculate_fan_in_and_fan_out(&[32, 16, 3, 3]),
        (16 * 9, 32 * 9)
    );
}

#[test]
fn test_gains() {
    assert!((calculate_gain(NonLinearity::Linear) - 1.0).abs() < 1e-5);
    assert!((calculate_gain(NonLinearity::ReLU) - 2.0f32.sqrt()).abs() < 1e-5);
    assert!((calculate_gain(NonLinearity::Tanh) - 5.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_xavier_initializations() {
    let x_uni = xavier_uniform(&[100, 200], 1.0);
    assert_eq!(x_uni.shape(), &[100, 200]);
    let slice_uni = x_uni.as_slice();
    let mean_uni: f32 = slice_uni.iter().sum::<f32>() / slice_uni.len() as f32;
    assert!(mean_uni.abs() < 0.05);

    let x_norm = xavier_normal(&[100, 200], 1.0);
    assert_eq!(x_norm.shape(), &[100, 200]);
    let slice_norm = x_norm.as_slice();
    let mean_norm: f32 = slice_norm.iter().sum::<f32>() / slice_norm.len() as f32;
    assert!(mean_norm.abs() < 0.05);
}

#[test]
fn test_kaiming_initializations() {
    let k_uni = kaiming_uniform(&[64, 128], 0.0, FanMode::FanIn, NonLinearity::ReLU);
    assert_eq!(k_uni.shape(), &[64, 128]);

    let k_norm = kaiming_normal(&[64, 128], 0.0, FanMode::FanIn, NonLinearity::ReLU);
    assert_eq!(k_norm.shape(), &[64, 128]);
}

#[test]
fn test_orthogonal_initialization() {
    // Orthogonal matrix [N, N]: Q^T * Q = I
    let n = 32;
    let q = orthogonal(&[n, n], 1.0).expect("Orthogonal init failed");
    assert_eq!(q.shape(), &[n, n]);

    let q_t = q.transpose(0, 1).unwrap();
    let ident = q_t.matmul(&q).unwrap();

    let slice = ident.as_slice();
    for i in 0..n {
        for j in 0..n {
            let val = slice[i * n + j];
            if i == j {
                assert!(
                    (val - 1.0).abs() < 1e-3,
                    "Diagonal element at ({}, {}) = {} (expected 1.0)",
                    i,
                    j,
                    val
                );
            } else {
                assert!(
                    val.abs() < 1e-3,
                    "Off-diagonal element at ({}, {}) = {} (expected 0.0)",
                    i,
                    j,
                    val
                );
            }
        }
    }
}
