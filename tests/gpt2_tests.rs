use neural_network_engine::prelude::*;

#[test]
fn test_gpt2_model_shapes_and_forward() {
    let config = GPT2Config {
        vocab_size: 150,
        max_position_embeddings: 64,
        d_model: 32,
        num_heads: 4,
        num_layers: 2,
        d_ff: 128,
        layer_norm_eps: 1e-5,
    };

    let model = GPT2Model::new(config);

    // Batch of 2, Sequence length of 6
    let input_raw = RawTensor::from_slice(
        &[
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // Batch 0
            5.0, 15.0, 25.0, 35.0, 45.0, 55.0, // Batch 1
        ],
        &[2, 6],
    );
    let input_ids = Tensor::new(input_raw, false);

    let logits = model.forward_tokens(&input_ids, 0, None).unwrap();
    assert_eq!(logits.shape(), &[2, 6, 150]);

    // Backward gradient flow check
    let loss = logits.sum_all();
    loss.backward();

    assert!(model.wte.weight.grad().is_some());
    assert!(model.wpe.weight.grad().is_some());
    assert!(model.blocks[0].ln_1.weight.grad().is_some());
    assert!(model.blocks[0].attn.q_proj.weight.grad().is_some());
    assert!(model.blocks[0].mlp_fc.weight.grad().is_some());
    assert!(model.lm_head.weight.grad().is_some());
}

#[test]
fn test_gpt2_kv_cache_equivalence() {
    let config = GPT2Config::tiny();
    let model = GPT2Model::new(config.clone());

    let tokens = [2, 5, 8, 12];
    let full_tensor = Tensor::from_slice(
        &tokens.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[1, tokens.len()],
        false,
    );

    // 1. Full non-cached pass
    let full_logits = model.forward_tokens(&full_tensor, 0, None).unwrap();
    let full_slice = full_logits.data();

    // 2. Incremental pass with KVCache
    let mut cache = KVCache::new(config.num_layers);
    let mut last_cached_logits: Option<RawTensor> = None;

    for (pos, &token) in tokens.iter().enumerate() {
        let single_tok = Tensor::scalar(token as f32, false)
            .reshape(&[1, 1])
            .unwrap();
        let step_logits = model
            .forward_tokens(&single_tok, pos, Some(&mut cache))
            .unwrap();
        last_cached_logits = Some(step_logits.data());
    }

    // Compare logits at the last position (pos = 3)
    let vocab_size = config.vocab_size;
    let full_last_row =
        &full_slice.as_slice()[(tokens.len() - 1) * vocab_size..tokens.len() * vocab_size];
    let cached_row = last_cached_logits.unwrap().as_slice().to_vec();

    for (a, b) in full_last_row.iter().zip(cached_row.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "Cached step logits deviate from full pass logits: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_gpt2_text_generation() {
    let config = GPT2Config::tiny();
    let model = GPT2Model::new(config);

    let prompt = vec![10, 20];
    let generated = model.generate(&prompt, 5, None).unwrap();
    assert_eq!(generated.len(), 7); // 2 prompt tokens + 5 generated tokens
    assert_eq!(&generated[..2], &prompt[..]);
}
