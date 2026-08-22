use neural_network_engine::prelude::*;

#[test]
fn test_bert_tri_embeddings_and_segment_encoding() {
    let config = BertConfig {
        vocab_size: 150,
        d_model: 32,
        num_layers: 2,
        num_heads: 2,
        d_ff: 64,
        max_position_embeddings: 64,
        type_vocab_size: 2,
        layer_norm_eps: 1e-6,
    };

    let bert = BertModel::new(config);

    // [CLS] Question [SEP] Context [SEP] -> SeqLen = 8
    let input_raw = RawTensor::from_slice(
        &[
            2.0, 10.0, 15.0, 3.0, 20.0, 25.0, 30.0, 3.0, // Batch 0
            2.0, 5.0, 8.0, 3.0, 12.0, 18.0, 22.0, 3.0, // Batch 1
        ],
        &[2, 8],
    );
    let input_ids = Tensor::new(input_raw, false);

    let type_raw = RawTensor::from_slice(
        &[
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // Batch 0
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // Batch 1
        ],
        &[2, 8],
    );
    let token_type_ids = Tensor::new(type_raw, false);

    let (seq_out, pooled_out) = bert
        .forward_bert(&input_ids, Some(&token_type_ids))
        .unwrap();
    assert_eq!(seq_out.shape(), &[2, 8, 32]);
    assert_eq!(pooled_out.shape(), &[2, 32]);

    // Backward pass
    let loss = seq_out.sum_all().add(&pooled_out.sum_all()).unwrap();
    loss.backward();

    assert!(bert.embeddings.word_embeddings.weight.grad().is_some());
    assert!(bert.embeddings.position_embeddings.weight.grad().is_some());
    assert!(bert
        .embeddings
        .token_type_embeddings
        .weight
        .grad()
        .is_some());
    assert!(bert.pooler.dense.weight.grad().is_some());
}

#[test]
fn test_bert_qa_head_forward_backward() {
    let config = BertConfig::tiny();
    let qa_model = BertForQuestionAnswering::new(config);

    let input_ids = Tensor::new(
        RawTensor::from_slice(&[2.0, 10.0, 3.0, 20.0, 30.0, 40.0, 3.0], &[1, 7]),
        false,
    );
    let type_ids = Tensor::new(
        RawTensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[1, 7]),
        false,
    );

    let (start_logits, end_logits) = qa_model.forward_qa(&input_ids, Some(&type_ids)).unwrap();

    assert_eq!(start_logits.shape(), &[1, 7]);
    assert_eq!(end_logits.shape(), &[1, 7]);

    let loss = start_logits.sum_all().add(&end_logits.sum_all()).unwrap();
    loss.backward();

    assert!(qa_model.qa_outputs.weight.grad().is_some());
}

#[test]
fn test_bert_sequence_embedding_and_cosine_similarity() {
    let config = BertConfig::tiny();
    let emb_model = BertForSequenceEmbedding::new(config);

    let input_ids = Tensor::new(
        RawTensor::from_slice(&[2.0, 15.0, 25.0, 35.0, 3.0], &[1, 5]),
        false,
    );

    let emb = emb_model.forward_embedding(&input_ids, None).unwrap();
    assert_eq!(emb.shape(), &[1, 64]);

    // Test cosine similarity utility
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0];
    let v4 = vec![-1.0, 0.0, 0.0];

    assert!((BertForSequenceEmbedding::cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
    assert!(BertForSequenceEmbedding::cosine_similarity(&v1, &v3).abs() < 1e-5);
    assert!((BertForSequenceEmbedding::cosine_similarity(&v1, &v4) - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_bert_sequence_length_exceeded_error() {
    let config = BertConfig {
        max_position_embeddings: 10,
        ..BertConfig::tiny()
    };
    let bert = BertModel::new(config);

    // Sequence length 15 exceeds max_position_embeddings=10
    let long_inputs = Tensor::new(RawTensor::zeros(&[1, 15]), false);
    let result = bert.forward_bert(&long_inputs, None);
    assert!(result.is_err());
}
