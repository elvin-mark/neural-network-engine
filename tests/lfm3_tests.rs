use neural_network_engine::autograd::NoGradGuard;
use neural_network_engine::models::lfm3::{Lfm3Cache, Lfm3Config, Lfm3LayerType, Lfm3Model};
use neural_network_engine::RawTensor;
use std::collections::HashMap;
use std::path::Path;

#[test]
fn lfm3_1_7b_config_matches_official_spec() {
    let cfg = Lfm3Config::lfm3_1_7b();
    assert_eq!(cfg.vocab_size, 128_000);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.intermediate_size, 7680);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.conv_l_cache, 7);
    assert!(!cfg.conv_bias);
    assert_eq!(cfg.sliding_window, 256);
    assert!(cfg.use_sandwich_norm);
    assert!(cfg.tie_word_embeddings);

    let conv_count = cfg
        .layer_types
        .iter()
        .filter(|&&lt| lt == Lfm3LayerType::Conv)
        .count();
    let full_count = cfg
        .layer_types
        .iter()
        .filter(|&&lt| lt == Lfm3LayerType::FullAttention)
        .count();
    let sliding_count = cfg
        .layer_types
        .iter()
        .filter(|&&lt| lt == Lfm3LayerType::SlidingAttention)
        .count();
    assert_eq!(conv_count, 13);
    assert_eq!(full_count, 6);
    assert_eq!(sliding_count, 5);
    assert_eq!(cfg.layer_types[0], Lfm3LayerType::Conv);
    assert_eq!(cfg.layer_types[2], Lfm3LayerType::FullAttention);
    assert_eq!(cfg.layer_types[4], Lfm3LayerType::SlidingAttention);
}

#[test]
fn lfm3_mini_forward_and_cached_parity() {
    let _guard = NoGradGuard::new();
    let config = Lfm3Config::mini(128, 64);
    let model = Lfm3Model::new(config.clone());

    // Length 6 > sliding_window (4), exercising both sliding window masking and conv1d width 7
    let tokens = [3usize, 17, 42, 9, 11, 25];
    let uncached = model.forward_tokens(&tokens, 1, tokens.len(), 0).unwrap();
    assert_eq!(uncached.shape(), vec![1, tokens.len(), 128]);

    let uncached_last = uncached
        .slice(1, tokens.len() - 1, tokens.len())
        .unwrap()
        .data()
        .to_contiguous();

    // Cached prefill on first 3 tokens, then decode remaining 3 tokens step-by-step
    let mut cache = Lfm3Cache::new(&config);
    let _ = model
        .forward_tokens_cached(&tokens[..3], 1, 3, 0, Some(&mut cache))
        .unwrap();
    let _ = model
        .forward_tokens_cached(&[tokens[3]], 1, 1, 3, Some(&mut cache))
        .unwrap();
    let _ = model
        .forward_tokens_cached(&[tokens[4]], 1, 1, 4, Some(&mut cache))
        .unwrap();
    let cached_step5 = model
        .forward_tokens_cached(&[tokens[5]], 1, 1, 5, Some(&mut cache))
        .unwrap();
    let cached_last = cached_step5.data().to_contiguous();

    let mut max_diff = 0.0f32;
    for (&a, &b) in uncached_last
        .as_slice()
        .iter()
        .zip(cached_last.as_slice().iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-4,
        "Cached vs uncached LFM3 logits diverged: max_diff = {}",
        max_diff
    );
}

#[test]
fn lfm3_mini_generate_cached_deterministic() {
    let _guard = NoGradGuard::new();
    let config = Lfm3Config::mini(64, 32);
    let model = Lfm3Model::new(config);

    let out1 = model.generate_cached(&[1, 2, 3], 4, 0.0).unwrap();
    let out2 = model.generate_cached(&[1, 2, 3], 4, 0.0).unwrap();
    assert_eq!(out1.len(), 7);
    assert_eq!(out1, out2);
}

#[test]
fn lfm3_load_weights_supports_hf_prefix_and_tied_embeddings() {
    let _guard = NoGradGuard::new();
    let config = Lfm3Config::mini(32, 16);
    let mut model = Lfm3Model::new(config.clone());

    let mut weights = HashMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        RawTensor::ones(&[32, 64]),
    );
    for (i, &lt) in config
        .layer_types
        .iter()
        .enumerate()
        .take(config.num_hidden_layers)
    {
        let p = format!("model.layers.{}", i);
        weights.insert(
            format!("{}.operator_norm.weight", p),
            RawTensor::ones(&[64]),
        );
        weights.insert(format!("{}.ffn_norm.weight", p), RawTensor::ones(&[64]));
        weights.insert(
            format!("{}.post_operator_norm.weight", p),
            RawTensor::ones(&[64]),
        );
        weights.insert(
            format!("{}.post_ffn_norm.weight", p),
            RawTensor::ones(&[64]),
        );
        weights.insert(
            format!("{}.feed_forward.w1.weight", p),
            RawTensor::zeros(&[128, 64]),
        );
        weights.insert(
            format!("{}.feed_forward.w2.weight", p),
            RawTensor::zeros(&[64, 128]),
        );
        weights.insert(
            format!("{}.feed_forward.w3.weight", p),
            RawTensor::zeros(&[128, 64]),
        );

        if lt == Lfm3LayerType::Conv {
            weights.insert(
                format!("{}.conv.in_proj.weight", p),
                RawTensor::zeros(&[192, 64]),
            );
            weights.insert(
                format!("{}.conv.conv.weight", p),
                RawTensor::zeros(&[64, 1, 7]),
            );
            weights.insert(
                format!("{}.conv.out_proj.weight", p),
                RawTensor::zeros(&[64, 64]),
            );
        } else {
            weights.insert(
                format!("{}.self_attn.q_proj.weight", p),
                RawTensor::zeros(&[64, 64]),
            );
            weights.insert(
                format!("{}.self_attn.k_proj.weight", p),
                RawTensor::zeros(&[32, 64]),
            );
            weights.insert(
                format!("{}.self_attn.v_proj.weight", p),
                RawTensor::zeros(&[32, 64]),
            );
            weights.insert(
                format!("{}.self_attn.o_proj.weight", p),
                RawTensor::zeros(&[64, 64]),
            );
            weights.insert(
                format!("{}.self_attn.q_norm.weight", p),
                RawTensor::ones(&[16]),
            );
            weights.insert(
                format!("{}.self_attn.k_norm.weight", p),
                RawTensor::ones(&[16]),
            );
            weights.insert(
                format!("{}.self_attn.gate_proj.weight", p),
                RawTensor::zeros(&[4, 64]),
            );
        }
    }
    weights.insert(
        "model.embedding_norm.weight".to_string(),
        RawTensor::ones(&[64]),
    );

    model.load_weights(&weights).unwrap();
    assert!((model.lm_head.weight.data().as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn lfm3_mini_pytorch_parity_if_present() {
    let fixture_dir = Path::new("/tmp/lfm3_mini_parity");
    if !fixture_dir.exists() {
        return;
    }
    let _guard = NoGradGuard::new();
    let config = Lfm3Config::mini(128, 64);
    let mut model = Lfm3Model::new(config);
    model
        .load_safetensors(fixture_dir.join("model.safetensors"))
        .unwrap();

    let ref_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir.join("reference.json")).unwrap())
            .unwrap();
    let input_ids: Vec<usize> = ref_json["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let expected_gen: Vec<usize> = ref_json["gen_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let rust_logits = model
        .forward_tokens(&input_ids, 1, input_ids.len(), 0)
        .unwrap();
    let rust_raw = rust_logits.data().to_contiguous();
    let rust_slice = rust_raw.as_slice();

    let pt_logits = ref_json["logits"].as_array().unwrap();
    let mut max_diff = 0.0f32;
    for (t, row) in pt_logits.iter().enumerate() {
        for (v, val) in row.as_array().unwrap().iter().enumerate() {
            let pt_val = val.as_f64().unwrap() as f32;
            let r_val = rust_slice[t * 128 + v];
            max_diff = max_diff.max((pt_val - r_val).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "PyTorch vs Rust LFM3 logits diverged: max_diff = {}",
        max_diff
    );

    let rust_gen = model.generate_cached(&input_ids, 3, 0.0).unwrap();
    assert_eq!(rust_gen, expected_gen, "LFM3 generated token IDs mismatch!");
}
