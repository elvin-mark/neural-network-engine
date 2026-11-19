use neural_network_engine::autograd::NoGradGuard;
use neural_network_engine::models::qwen3_5::{
    Qwen3_5Cache, Qwen3_5Config, Qwen3_5LayerType, Qwen3_5Model, Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
};
use neural_network_engine::nn::module::Module;
use neural_network_engine::{RawTensor, Tensor};
use std::collections::HashMap;
use std::path::Path;

#[test]
fn qwen3_5_0_8b_config_matches_official_spec() {
    let cfg = Qwen3_5Config::qwen3_5_0_8b();
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.intermediate_size, 3584);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_num_value_heads, 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert!(cfg.tie_word_embeddings);

    let full_count = cfg
        .layer_types
        .iter()
        .filter(|&&lt| lt == Qwen3_5LayerType::FullAttention)
        .count();
    let linear_count = cfg
        .layer_types
        .iter()
        .filter(|&&lt| lt == Qwen3_5LayerType::LinearAttention)
        .count();
    assert_eq!(full_count, 6);
    assert_eq!(linear_count, 18);
    assert_eq!(cfg.layer_types[3], Qwen3_5LayerType::FullAttention);
    assert_eq!(cfg.layer_types[0], Qwen3_5LayerType::LinearAttention);
}

#[test]
fn qwen3_5_rmsnorm_is_zero_centered() {
    let _guard = NoGradGuard::new();
    let norm = Qwen3_5RMSNorm::new(4, 1e-6);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], false);

    // With weight = 0, multiplier is (1 + 0) = 1.0
    let y0 = norm.forward(&x).unwrap();
    let rms = ((1.0f32 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6).sqrt();
    let y0_slice = y0.data().to_contiguous();
    for (i, &val) in y0_slice.as_slice().iter().enumerate() {
        let expected = (i + 1) as f32 / rms;
        assert!(
            (val - expected).abs() < 1e-5,
            "idx {}: {} vs {}",
            i,
            val,
            expected
        );
    }
}

#[test]
fn qwen3_5_rmsnorm_gated_applies_silu_gate() {
    let _guard = NoGradGuard::new();
    let norm = Qwen3_5RMSNormGated::new(2, 1e-6);
    let x = Tensor::from_slice(&[3.0, 4.0], &[1, 2], false);
    let gate = Tensor::from_slice(&[0.0, 2.0], &[1, 2], false);

    let y = norm.forward(&x, &gate).unwrap();
    let y_raw = y.data().to_contiguous();
    let s = y_raw.as_slice();
    let rms = ((9.0f32 + 16.0) / 2.0 + 1e-6).sqrt();
    // silu(0) = 0, silu(2) = 2 / (1 + exp(-2))
    assert!(s[0].abs() < 1e-6);
    let expected_1 = (4.0 / rms) * (2.0 / (1.0 + (-2.0f32).exp()));
    assert!((s[1] - expected_1).abs() < 1e-5);
}

#[test]
fn qwen3_5_mini_forward_and_cached_parity() {
    let _guard = NoGradGuard::new();
    let config = Qwen3_5Config::mini(128, 64);
    let model = Qwen3_5Model::new(config.clone());

    let tokens = [3usize, 17, 42, 9, 11];
    let uncached = model.forward_tokens(&tokens, 1, tokens.len(), 0).unwrap();
    assert_eq!(uncached.shape(), vec![1, tokens.len(), 128]);

    let uncached_last = uncached
        .slice(1, tokens.len() - 1, tokens.len())
        .unwrap()
        .data()
        .to_contiguous();

    // Cached prefill on first 3 tokens, then step-by-step decode on last 2 tokens
    let mut cache = Qwen3_5Cache::new(&config);
    let _ = model
        .forward_tokens_cached(&tokens[..3], 1, 3, 0, Some(&mut cache))
        .unwrap();
    let _ = model
        .forward_tokens_cached(&[tokens[3]], 1, 1, 3, Some(&mut cache))
        .unwrap();
    let cached_step4 = model
        .forward_tokens_cached(&[tokens[4]], 1, 1, 4, Some(&mut cache))
        .unwrap();
    let cached_last = cached_step4.data().to_contiguous();

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
        "Cached vs uncached logits diverged: max_diff = {}",
        max_diff
    );
}

#[test]
fn qwen3_5_mini_generate_cached_deterministic() {
    let _guard = NoGradGuard::new();
    let config = Qwen3_5Config::mini(64, 32);
    let model = Qwen3_5Model::new(config);

    let out1 = model.generate_cached(&[1, 2, 3], 4, 0.0).unwrap();
    let out2 = model.generate_cached(&[1, 2, 3], 4, 0.0).unwrap();
    assert_eq!(out1.len(), 7);
    assert_eq!(out1, out2);
}

#[test]
fn qwen3_5_load_weights_supports_hf_prefix_and_tied_embeddings() {
    let _guard = NoGradGuard::new();
    let config = Qwen3_5Config::mini(32, 16);
    let mut model = Qwen3_5Model::new(config.clone());

    let mut weights = HashMap::new();
    let embed = RawTensor::ones(&[32, 64]);
    weights.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        embed,
    );
    for (i, &layer_type) in config
        .layer_types
        .iter()
        .enumerate()
        .take(config.num_hidden_layers)
    {
        let p = format!("model.language_model.layers.{}", i);
        weights.insert(
            format!("{}.input_layernorm.weight", p),
            RawTensor::zeros(&[64]),
        );
        weights.insert(
            format!("{}.post_attention_layernorm.weight", p),
            RawTensor::zeros(&[64]),
        );
        weights.insert(
            format!("{}.mlp.gate_proj.weight", p),
            RawTensor::zeros(&[128, 64]),
        );
        weights.insert(
            format!("{}.mlp.up_proj.weight", p),
            RawTensor::zeros(&[128, 64]),
        );
        weights.insert(
            format!("{}.mlp.down_proj.weight", p),
            RawTensor::zeros(&[64, 128]),
        );
        if layer_type == Qwen3_5LayerType::LinearAttention {
            weights.insert(
                format!("{}.linear_attn.in_proj_qkv.weight", p),
                RawTensor::zeros(&[192, 64]),
            );
            weights.insert(
                format!("{}.linear_attn.in_proj_z.weight", p),
                RawTensor::zeros(&[64, 64]),
            );
            weights.insert(
                format!("{}.linear_attn.in_proj_b.weight", p),
                RawTensor::zeros(&[4, 64]),
            );
            weights.insert(
                format!("{}.linear_attn.in_proj_a.weight", p),
                RawTensor::zeros(&[4, 64]),
            );
            weights.insert(
                format!("{}.linear_attn.conv1d.weight", p),
                RawTensor::zeros(&[192, 1, 4]),
            );
            weights.insert(format!("{}.linear_attn.dt_bias", p), RawTensor::ones(&[4]));
            weights.insert(format!("{}.linear_attn.A_log", p), RawTensor::zeros(&[4]));
            weights.insert(
                format!("{}.linear_attn.norm.weight", p),
                RawTensor::ones(&[16]),
            );
            weights.insert(
                format!("{}.linear_attn.out_proj.weight", p),
                RawTensor::zeros(&[64, 64]),
            );
        } else {
            weights.insert(
                format!("{}.self_attn.q_proj.weight", p),
                RawTensor::zeros(&[128, 64]),
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
                RawTensor::zeros(&[16]),
            );
            weights.insert(
                format!("{}.self_attn.k_norm.weight", p),
                RawTensor::zeros(&[16]),
            );
        }
    }
    weights.insert(
        "model.language_model.norm.weight".to_string(),
        RawTensor::zeros(&[64]),
    );

    model.load_weights(&weights).unwrap();
    // Tied lm_head should now be ones
    assert!((model.lm_head.weight.data().as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn qwen3_5_mini_pytorch_parity_if_present() {
    let fixture_dir = Path::new("/tmp/qwen3_5_mini_parity");
    if !fixture_dir.exists() {
        return;
    }
    let _guard = NoGradGuard::new();
    let config = Qwen3_5Config::mini(128, 64);
    let mut model = Qwen3_5Model::new(config);
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
        "PyTorch vs Rust Qwen3.5 logits diverged: max_diff = {}",
        max_diff
    );

    let rust_gen = model.generate_cached(&input_ids, 3, 0.0).unwrap();
    assert_eq!(rust_gen, expected_gen, "Generated token IDs mismatch!");
}
