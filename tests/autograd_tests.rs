use neural_network_engine::prelude::*;

#[test]
fn test_basic_autograd() {
    let a = Tensor::new(RawTensor::from_slice(&[2.0, 3.0], &[2]), true);
    let b = Tensor::new(RawTensor::from_slice(&[4.0, 5.0], &[2]), true);

    // c = a * b + a
    // dc/da = b + 1 = [5, 6]
    // dc/db = a = [2, 3]
    let ab = a.mul(&b).unwrap();
    let c = ab.add(&a).unwrap();
    let loss = c.sum_all();

    loss.backward();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.as_slice(), &[5.0, 6.0]);
    assert_eq!(grad_b.as_slice(), &[2.0, 3.0]);
}

#[test]
fn test_branching_and_accumulation() {
    // y = x^2 + 2*x + 1
    // dy/dx = 2*x + 2
    let x = Tensor::scalar(3.0, true);
    let x2 = x.mul(&x).unwrap();
    let two_x = x.mul(&Tensor::scalar(2.0, false)).unwrap();
    let y = x2
        .add(&two_x)
        .unwrap()
        .add(&Tensor::scalar(1.0, false))
        .unwrap();

    y.backward();

    let grad_x = x.grad().unwrap();
    assert_eq!(grad_x.item(), 8.0); // 2*3 + 2 = 8
}

#[test]
fn test_matmul_backward() {
    let a = Tensor::new(RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let b = Tensor::new(RawTensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]), true);

    let c = a.matmul(&b).unwrap();
    let loss = c.sum_all();
    loss.backward();

    // dL/dA = 1 * B^T = [[5, 7], [5, 7]] + [[6, 8], [6, 8]] = [[11, 15], [11, 15]]
    let grad_a = a.grad().unwrap();
    assert_eq!(grad_a.as_slice(), &[11.0, 15.0, 11.0, 15.0]);

    // dL/dB = A^T * 1 = [[1, 3], [2, 4]] * [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
    let grad_b = b.grad().unwrap();
    assert_eq!(grad_b.as_slice(), &[4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_no_grad_mode() {
    let a = Tensor::scalar(5.0, true);
    let b = Tensor::scalar(3.0, true);

    let c = no_grad(|| a.mul(&b).unwrap());

    assert!(!c.requires_grad());
}
