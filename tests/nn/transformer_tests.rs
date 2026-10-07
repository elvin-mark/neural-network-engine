use neural_network_engine::prelude::*;

#[test]
fn test_multi_head_attention_forward_backward() {
    let mha = MultiHeadAttention::new(32, 4, true);
    let x = Tensor::randn(&[2, 8, 32], 0.0, 1.0, true);

    let out = mha.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 8, 32]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &[2, 8, 32]);
    assert!(mha.q_proj.weight.grad().is_some());
    assert!(mha.k_proj.weight.grad().is_some());
    assert!(mha.v_proj.weight.grad().is_some());
    assert!(mha.out_proj.weight.grad().is_some());
}

#[test]
fn test_transformer_block_forward_backward() {
    let block = TransformerBlock::new(32, 4, true);
    let x = Tensor::randn(&[2, 6, 32], 0.0, 1.0, true);

    let out = block.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 6, 32]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &[2, 6, 32]);
}

#[test]
fn test_transformer_lm_forward_backward() {
    let model = TransformerLM::new(50, 16, 32, 4, 2);
    let token_ids = vec![1, 5, 10, 20, 30, 42]; // [1, 6]

    let logits = model.forward_tokens(&token_ids, 1, 6).unwrap();
    assert_eq!(logits.shape(), &[1, 6, 50]);

    let loss = logits.sum_all();
    loss.backward();

    assert!(model.tok_emb.weight.grad().is_some());
    assert!(model.pos_emb.weight.grad().is_some());
    assert!(model.lm_head.weight.grad().is_some());
}
