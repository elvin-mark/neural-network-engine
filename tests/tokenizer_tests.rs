use neural_network_engine::prelude::*;

#[test]
fn test_byte_level_bpe_training_and_compression() {
    let corpus = "Once upon a time, there was a little girl named Lily. Lily loved to explore the bright green garden. Once upon a time, Lily found a golden key.";
    let tokenizer = ByteLevelBPE::train(corpus, 280, &["<s>", "</s>"]).unwrap();

    assert_eq!(tokenizer.vocab_size(), 280);
    assert_eq!(tokenizer.bos_token_id(), Some(0));
    assert_eq!(tokenizer.eos_token_id(), Some(1));

    let encoded = tokenizer.encode(corpus).unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();

    assert_eq!(decoded, corpus);
    // Compression check: encoded tokens must be fewer than raw bytes
    assert!(encoded.len() < corpus.len());
}

#[test]
fn test_byte_level_bpe_unicode_utf8_lossless() {
    let text = "Rust 🦀 Neural Engine | UTF-8: こんにちは, Bonjour, Grüß Gott, Привет мир! ✨🚀";
    let tokenizer = ByteLevelBPE::default();

    let encoded = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_special_tokens_handling() {
    let special_tokens = &["<unk>", "<s>", "</s>", "<pad>"];
    let tokenizer = ByteLevelBPE::with_special_tokens(special_tokens);

    assert_eq!(tokenizer.unk_token_id(), Some(0));
    assert_eq!(tokenizer.bos_token_id(), Some(1));
    assert_eq!(tokenizer.eos_token_id(), Some(2));
    assert_eq!(tokenizer.pad_token_id(), Some(3));

    let text = "<s> Hello World </s>";
    let encoded = tokenizer.encode(text).unwrap();

    assert_eq!(encoded.first(), Some(&1));
    assert_eq!(encoded.last(), Some(&2));

    let decoded = tokenizer.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_json_serialization_roundtrip() {
    let text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps again!";
    let tokenizer = ByteLevelBPE::train(text, 275, &["<unk>", "<s>", "</s>"]).unwrap();

    let temp_dir = std::env::temp_dir();
    let json_path = temp_dir.join("test_tokenizer_export.json");

    tokenizer.save_json(&json_path).unwrap();
    let loaded = ByteLevelBPE::load_json(&json_path).unwrap();

    assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
    assert_eq!(tokenizer.bos_token_id(), loaded.bos_token_id());

    let encoded_orig = tokenizer.encode(text).unwrap();
    let encoded_loaded = loaded.encode(text).unwrap();
    assert_eq!(encoded_orig, encoded_loaded);

    let decoded = loaded.decode(&encoded_loaded).unwrap();
    assert_eq!(decoded, text);

    let _ = std::fs::remove_file(json_path);
}

#[test]
fn test_oov_resilience() {
    // Train on a simple corpus
    let train_text = "cat dog bird fish lion tiger elephant";
    let tokenizer = ByteLevelBPE::train(train_text, 270, &["<s>", "</s>"]).unwrap();

    // Encode completely unseen vocabulary and foreign scripts
    let novel_text = "Quantum computing and 宇宙航空研究開発機構 are fascinating!";
    let encoded = tokenizer.encode(novel_text).unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();

    assert_eq!(decoded, novel_text);
}

#[test]
fn test_out_of_bounds_token_error() {
    let tokenizer = ByteLevelBPE::default();
    let invalid_token_id = 999999;

    let result = tokenizer.decode(&[invalid_token_id]);
    assert!(result.is_err());
    if let Err(EngineError::TokenizerError(msg)) = result {
        assert!(msg.contains("out of vocabulary range"));
    } else {
        panic!("Expected EngineError::TokenizerError");
    }
}
