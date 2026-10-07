use neural_network_engine::prelude::*;

#[test]
fn test_modern_bert_rotary_attention_and_shapes() {
    let config = ModernBertConfig {
        vocab_size: 120,
        d_model: 32,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 128,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    };

    let model = ModernBertModel::new(config);

    // Batch of 2 sequences, each of length 8
    let input_raw = RawTensor::from_slice(
        &[
            1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 2.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0,
            72.0,
        ],
        &[2, 8],
    );
    let input_ids = Tensor::new(input_raw, false);

    let (seq_out, pooled_out) = model.forward_model(&input_ids).unwrap();
    assert_eq!(seq_out.shape(), &[2, 8, 32]);
    assert_eq!(pooled_out.shape(), &[2, 32]);

    // Verify autograd backward flow through RoPE, Pre-RMSNorm, and GeGLU MLP
    let loss = seq_out.sum_all().add(&pooled_out.sum_all()).unwrap();
    loss.backward();

    assert!(model.embeddings.weight.grad().is_some());
    assert!(model.encoder.layers[0]
        .attention
        .q_proj
        .weight
        .grad()
        .is_some());
    assert!(model.encoder.layers[0]
        .attention
        .out_proj
        .weight
        .grad()
        .is_some());
    assert!(model.encoder.layers[0]
        .mlp
        .gate_proj
        .weight
        .grad()
        .is_some());
    assert!(model.encoder.layers[0]
        .mlp
        .down_proj
        .weight
        .grad()
        .is_some());
}

#[test]
fn test_modern_bert_sequence_classification() {
    let config = ModernBertConfig::tiny();
    let num_classes = 4;
    let classifier_model = ModernBertForSequenceClassification::new(config, num_classes);

    let input_raw = RawTensor::from_slice(&[2.0, 4.0, 6.0, 8.0], &[1, 4]);
    let input_ids = Tensor::new(input_raw, false);

    let logits = classifier_model.forward_classification(&input_ids).unwrap();
    assert_eq!(logits.shape(), &[1, 4]);

    let loss = logits.sum_all();
    loss.backward();
    assert!(classifier_model.classifier.weight.grad().is_some());
}

#[test]
fn test_modern_bert_sequence_embedding_unit_norm() {
    let config = ModernBertConfig::tiny();
    let emb_model = ModernBertForSequenceEmbedding::new(config);

    let input_raw = RawTensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );
    let input_ids = Tensor::new(input_raw, false);

    let embeddings = emb_model.forward_embedding(&input_ids).unwrap();
    assert_eq!(embeddings.shape(), &[2, 64]);

    // Verify each embedding vector has unit L2 norm (|v| = 1.0)
    let raw = embeddings.data();
    let slice = raw.as_slice();
    for b in 0..2 {
        let row = &slice[b * 64..(b + 1) * 64];
        let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Row {} norm expected ~1.0, got {}",
            b,
            norm
        );
    }
}
