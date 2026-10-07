use neural_network_engine::prelude::*;

#[test]
fn test_kv_cache_basic_operations() {
    let mut cache = KVCache::new(4);
    assert_eq!(cache.num_layers(), 4);
    assert_eq!(cache.current_seq_len(0), 0);

    let k1 = Tensor::randn(&[2, 4, 3, 16], 0.0, 1.0, false); // [B=2, H=4, T=3, D=16]
    let v1 = Tensor::randn(&[2, 4, 3, 16], 0.0, 1.0, false);

    let (k_all, v_all) = cache.update(0, &k1, &v1).unwrap();
    assert_eq!(k_all.shape(), &[2, 4, 3, 16]);
    assert_eq!(v_all.shape(), &[2, 4, 3, 16]);
    assert_eq!(cache.current_seq_len(0), 3);

    // Append 1 more token
    let k2 = Tensor::randn(&[2, 4, 1, 16], 0.0, 1.0, false);
    let v2 = Tensor::randn(&[2, 4, 1, 16], 0.0, 1.0, false);
    let (k_all2, v_all2) = cache.update(0, &k2, &v2).unwrap();
    assert_eq!(k_all2.shape(), &[2, 4, 4, 16]);
    assert_eq!(v_all2.shape(), &[2, 4, 4, 16]);
    assert_eq!(cache.current_seq_len(0), 4);

    cache.reset();
    assert_eq!(cache.current_seq_len(0), 0);
}

#[test]
fn test_multihead_attention_cached_parity() {
    let mha = MultiHeadAttention::new(32, 4, true);
    let x = Tensor::randn(&[1, 5, 32], 0.0, 1.0, false);

    // 1. Standard full-sequence forward pass
    let full_out = mha.forward_attention(&x).unwrap();

    // 2. Step-by-step cached forward pass
    let mut cache_slot = (Tensor::zeros(&[0], false), Tensor::zeros(&[0], false));
    let mut step_outputs = Vec::new();

    for t in 0..5 {
        let x_step = x.slice(1, t, t + 1).unwrap(); // [1, 1, 32]
        let out_step = mha
            .forward_attention_cached(&x_step, Some(&mut cache_slot))
            .unwrap();
        step_outputs.push(out_step);
    }

    let step_refs: Vec<&Tensor> = step_outputs.iter().collect();
    let cached_out = Tensor::cat(&step_refs, 1).unwrap();

    // Verify shapes and numerical parity between cached decoding and full sequence
    assert_eq!(full_out.shape(), cached_out.shape());
    let full_slice = full_out.data().to_contiguous();
    let cached_slice = cached_out.data().to_contiguous();

    for (a, b) in full_slice.as_slice().iter().zip(cached_slice.as_slice()) {
        assert!(
            (a - b).abs() < 1e-4,
            "MHA cached parity mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_llama_gqa_cached_parity() {
    let config = LlamaConfig::mini(100, 32);
    let gqa = GroupedQueryAttention::new(&config);
    let x = Tensor::randn(&[1, 6, config.d_model], 0.0, 1.0, false);

    // 1. Full sequence forward pass
    let full_out = gqa.forward_gqa(&x, 0).unwrap();

    // 2. Step-by-step cached forward pass
    let mut cache_slot = (Tensor::zeros(&[0], false), Tensor::zeros(&[0], false));
    let mut step_outputs = Vec::new();

    for t in 0..6 {
        let x_step = x.slice(1, t, t + 1).unwrap(); // [1, 1, d_model]
        let out_step = gqa
            .forward_gqa_cached(&x_step, t, Some(&mut cache_slot))
            .unwrap();
        step_outputs.push(out_step);
    }

    let step_refs: Vec<&Tensor> = step_outputs.iter().collect();
    let cached_out = Tensor::cat(&step_refs, 1).unwrap();

    assert_eq!(full_out.shape(), cached_out.shape());
    let full_slice = full_out.data().to_contiguous();
    let cached_slice = cached_out.data().to_contiguous();

    for (a, b) in full_slice.as_slice().iter().zip(cached_slice.as_slice()) {
        assert!(
            (a - b).abs() < 1e-4,
            "GQA cached parity mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_llama2_lm_generate_cached() {
    let config = LlamaConfig::mini(50, 32);
    let llama = Llama2LM::new(config);

    let prompt = vec![1, 15, 23];
    let generated = llama.generate_cached(&prompt, 10, 0.0).unwrap();

    assert_eq!(generated.len(), prompt.len() + 10);
    assert_eq!(&generated[..prompt.len()], &prompt);
}
