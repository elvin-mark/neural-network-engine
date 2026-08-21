use neural_network_engine::prelude::*;

#[test]
fn test_gradcheck_arithmetic() {
    let x = Tensor::randn(&[3, 4], 1.0, 0.5, true);

    // f(x) = sum(x^3 - 2*x + 5)
    let err = gradcheck(
        |t| {
            let x3 = t.powf(3.0)?;
            let two_x = t.mul(&Tensor::scalar(2.0, false))?;
            let sub = x3.sub(&two_x)?;
            let res = sub.add(&Tensor::scalar(5.0, false))?;
            Ok(res.sum_all())
        },
        &x,
        1e-3,
        1e-3,
    )
    .unwrap();

    assert!(err < 1e-3, "Relative error: {}", err);
}

#[test]
fn test_gradcheck_activations() {
    let x = Tensor::randn(&[4, 4], 0.0, 1.0, true);

    // Sigmoid gradcheck
    let err_sig = gradcheck(|t| Ok(t.sigmoid()?.sum_all()), &x, 1e-3, 1e-3).unwrap();
    assert!(err_sig < 1e-3);

    // Tanh gradcheck
    let err_tanh = gradcheck(|t| Ok(t.tanh()?.sum_all()), &x, 1e-3, 1e-3).unwrap();
    assert!(err_tanh < 1e-3);

    // GELU gradcheck
    let err_gelu = gradcheck(|t| Ok(t.gelu()?.sum_all()), &x, 1e-3, 1e-3).unwrap();
    assert!(err_gelu < 1e-3);
}

#[test]
fn test_gradcheck_matmul() {
    let x = Tensor::randn(&[3, 4], 0.0, 1.0, true);
    let w = Tensor::randn(&[4, 5], 0.0, 1.0, false);

    let err = gradcheck(
        |t| {
            let out = t.matmul(&w)?;
            Ok(out.sum_all())
        },
        &x,
        1e-3,
        1e-3,
    )
    .unwrap();

    assert!(err < 1e-3);
}

#[test]
fn test_gradcheck_log_softmax() {
    let x = Tensor::randn(&[3, 5], 0.0, 1.0, true);

    let err = gradcheck(
        |t| {
            let ls = t.log_softmax(1)?;
            Ok(ls.sum_all())
        },
        &x,
        1e-3,
        1e-3,
    )
    .unwrap();

    assert!(err < 1e-3);
}

#[test]
fn test_gradcheck_conv2d() {
    let x = Tensor::randn(&[2, 2, 6, 6], 0.0, 1.0, true);
    let w = Tensor::randn(&[3, 2, 3, 3], 0.0, 1.0, false);
    let params = Conv2dParams::default();

    let err = gradcheck(
        |t| {
            let out = t.conv2d(&w, None, params)?;
            Ok(out.sum_all())
        },
        &x,
        1e-3,
        1e-3,
    )
    .unwrap();

    assert!(err < 1e-3);
}

#[test]
fn test_gradcheck_layernorm() {
    let x = Tensor::randn(&[2, 8], 0.0, 1.0, true);
    let weights = Tensor::randn(&[2, 8], 0.0, 1.0, false);
    let ln = LayerNorm::new(8);

    let err = gradcheck(
        |t| {
            let out = ln.forward(t)?;
            let weighted = out.mul(&weights)?;
            Ok(weighted.sum_all())
        },
        &x,
        1e-3,
        1e-3,
    )
    .unwrap();

    assert!(err < 1e-3);
}
