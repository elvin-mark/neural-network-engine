use neural_network_engine::prelude::*;

#[test]
fn test_rmsnorm_forward_backward() {
    let norm = RMSNorm::new(16);
    let x = Tensor::randn(&[2, 4, 16], 0.0, 1.0, true);

    let out = norm.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 16]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(norm.weight.grad().is_some());
}

#[test]
fn test_rotary_embedding() {
    let rope = RotaryEmbedding::new(16, 32, 10000.0);
    let q = Tensor::randn(&[2, 4, 8, 16], 0.0, 1.0, true);

    let q_rot = rope.apply(&q, 0).unwrap();
    assert_eq!(q_rot.shape(), &[2, 4, 8, 16]);

    let loss = q_rot.sum_all();
    loss.backward();

    assert!(q.grad().is_some());
}

#[test]
fn test_gqa_forward_backward() {
    let config = LlamaConfig {
        vocab_size: 50,
        d_model: 32,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 2, // GQA G=2
        num_layers: 1,
        max_seq_len: 16,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    let gqa = GroupedQueryAttention::new(&config);
    let x = Tensor::randn(&[2, 6, 32], 0.0, 1.0, true);

    let out = gqa.forward_gqa(&x, 0).unwrap();
    assert_eq!(out.shape(), &[2, 6, 32]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(gqa.q_proj.weight.grad().is_some());
    assert!(gqa.k_proj.weight.grad().is_some());
    assert!(gqa.v_proj.weight.grad().is_some());
    assert!(gqa.o_proj.weight.grad().is_some());
}

#[test]
fn test_swiglu_forward_backward() {
    let swiglu = SwiGLU::new(32, 64);
    let x = Tensor::randn(&[2, 6, 32], 0.0, 1.0, true);

    let out = swiglu.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 6, 32]);

    let loss = out.sum_all();
    loss.backward();

    assert!(x.grad().is_some());
    assert!(swiglu.gate_proj.weight.grad().is_some());
    assert!(swiglu.up_proj.weight.grad().is_some());
    assert!(swiglu.down_proj.weight.grad().is_some());
}

#[test]
fn test_llama2_lm_forward_backward() {
    let config = LlamaConfig::mini(60, 16);
    let model = Llama2LM::new(config);

    let tokens = vec![1, 10, 25, 42, 55, 3];
    let logits = model.forward_tokens(&tokens, 1, 6, 0).unwrap();
    assert_eq!(logits.shape(), &[1, 6, 60]);

    let loss = logits.sum_all();
    loss.backward();

    assert!(model.tok_embeddings.weight.grad().is_some());
    assert!(model.lm_head.weight.grad().is_some());
    assert!(model.norm.weight.grad().is_some());
}
