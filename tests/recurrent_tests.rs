use neural_network_engine::prelude::*;

#[test]
fn test_rnn_cell_forward_backward() {
    let cell = RNNCell::new(8, 16, RNNActivation::Tanh);
    let x = Tensor::randn(&[4, 8], 0.0, 1.0, true);
    let h0 = Tensor::zeros(&[4, 16], true);

    let h1 = cell.forward_step(&x, Some(&h0)).unwrap();
    assert_eq!(h1.shape(), &[4, 16]);

    let loss = h1.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(h0.grad().is_some());
    assert!(cell.weight_ih.grad().is_some());
    assert!(cell.weight_hh.grad().is_some());
}

#[test]
fn test_rnn_sequence_unidirectional_and_bidirectional() {
    // 1. Unidirectional RNN
    let rnn = RNN::new(10, 20, 2, RNNActivation::Tanh, false, 0.0);
    let x = Tensor::randn(&[2, 5, 10], 0.0, 1.0, true); // [batch=2, seq_len=5, input_size=10]

    let (out, h_n) = rnn.forward_seq(&x, None).unwrap();
    assert_eq!(out.shape(), &[2, 5, 20]);
    assert_eq!(h_n.shape(), &[2, 2, 20]); // [num_layers=2, batch=2, hidden=20]

    let loss = out.sum_all();
    loss.backward();
    assert!(x.grad().is_some());

    // 2. Bidirectional RNN
    let bi_rnn = RNN::new(10, 20, 1, RNNActivation::ReLU, true, 0.0);
    let (bi_out, bi_hn) = bi_rnn.forward_seq(&x, None).unwrap();
    assert_eq!(bi_out.shape(), &[2, 5, 40]); // [batch=2, seq=5, 2 * hidden=40]
    assert_eq!(bi_hn.shape(), &[2, 2, 20]); // [num_directions=2, batch=2, hidden=20]
}

#[test]
fn test_lstm_cell_forward_backward() {
    let cell = LSTMCell::new(12, 24);
    let x = Tensor::randn(&[3, 12], 0.0, 1.0, true);
    let h0 = Tensor::zeros(&[3, 24], true);
    let c0 = Tensor::zeros(&[3, 24], true);

    let (h1, c1) = cell.forward_step(&x, Some((&h0, &c0))).unwrap();
    assert_eq!(h1.shape(), &[3, 24]);
    assert_eq!(c1.shape(), &[3, 24]);

    let loss = h1.add(&c1).unwrap().sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(h0.grad().is_some());
    assert!(c0.grad().is_some());
    assert!(cell.weight_ih.grad().is_some());
    assert!(cell.weight_hh.grad().is_some());
}

#[test]
fn test_lstm_sequence_multi_layer_and_bidirectional() {
    // 2-layer Bidirectional LSTM
    let lstm = LSTM::new(16, 32, 2, true, 0.1);
    let x = Tensor::randn(&[4, 6, 16], 0.0, 1.0, true); // [B=4, T=6, D=16]

    let (out, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();
    assert_eq!(out.shape(), &[4, 6, 64]); // 2 directions * 32 hidden = 64
    assert_eq!(h_n.shape(), &[4, 4, 32]); // 2 layers * 2 directions = 4
    assert_eq!(c_n.shape(), &[4, 4, 32]);

    let loss = out.sum_all();
    loss.backward();
    assert!(x.grad().is_some());
}

#[test]
fn test_gru_cell_forward_backward() {
    let cell = GRUCell::new(10, 20);
    let x = Tensor::randn(&[2, 10], 0.0, 1.0, true);
    let h0 = Tensor::zeros(&[2, 20], true);

    let h1 = cell.forward_step(&x, Some(&h0)).unwrap();
    assert_eq!(h1.shape(), &[2, 20]);

    let loss = h1.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(h0.grad().is_some());
    assert!(cell.weight_ih.grad().is_some());
    assert!(cell.weight_hh.grad().is_some());
}

#[test]
fn test_gru_sequence_multi_layer_and_bidirectional() {
    let gru = GRU::new(8, 16, 2, true, 0.0);
    let x = Tensor::randn(&[3, 7, 8], 0.0, 1.0, true);

    let (out, h_n) = gru.forward_seq(&x, None).unwrap();
    assert_eq!(out.shape(), &[3, 7, 32]); // 2 directions * 16 hidden = 32
    assert_eq!(h_n.shape(), &[4, 3, 16]); // 2 layers * 2 directions = 4

    let loss = out.sum_all();
    loss.backward();
    assert!(x.grad().is_some());
}

#[test]
fn test_lstm_sequence_learning_convergence() {
    // Train a small LSTM on a sequence prediction task
    let lstm = LSTM::new(4, 8, 1, false, 0.0);
    let fc = Linear::new(8, 1);
    let mut params = lstm.parameters();
    params.extend(fc.parameters());

    let mut optimizer = Adam::new(params, 0.05);

    // Target is cumulative sum indicator
    let x_data = RawTensor::randn(&[4, 5, 4], 0.0, 1.0);
    let y_data = RawTensor::randn(&[4, 1], 0.0, 1.0);

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for epoch in 0..15 {
        let x = Tensor::new(x_data.clone(), false);
        let y = Tensor::new(y_data.clone(), false);

        let (_out, (h_n, _)) = lstm.forward_seq(&x, None).unwrap();
        let last_h = h_n.squeeze(0).unwrap(); // [batch=4, hidden=8]
        let pred = fc.forward(&last_h).unwrap(); // [batch=4, 1]

        let loss = MSELoss::forward(&pred, &y).unwrap();
        let loss_val = loss.data().item();

        if epoch == 0 {
            initial_loss = loss_val;
        }
        if epoch == 14 {
            final_loss = loss_val;
        }

        optimizer.zero_grad();
        loss.backward();
        clip_grad_norm(&lstm.parameters(), 1.0);
        optimizer.step().unwrap();
    }

    assert!(
        final_loss < initial_loss,
        "LSTM loss did not decrease: initial {} vs final {}",
        initial_loss,
        final_loss
    );
}
