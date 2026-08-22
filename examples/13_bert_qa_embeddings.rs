//! Example 13: BERT for Question Answering Span Extraction and Semantic Text Embeddings.
//!
//! Workflow:
//! 1. Build Extractive QA & Semantic Similarity datasets.
//! 2. Tokenize `[CLS] Question [SEP] Context [SEP]` with Byte-Level BPE & segment IDs.
//! 3. Train `BertForQuestionAnswering` with joint start/end span Cross-Entropy loss.
//! 4. Perform extractive question answering inference on unseen queries.
//! 5. Extract semantic text embeddings and rank document candidates via Cosine Similarity.
//! 6. Save model weights in SafeTensors format.

use neural_network_engine::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 13_bert_qa_embeddings: BERT QA & Text Embedding Pipeline   ");
    println!("============================================================\n");

    // ---------------------------------------------------------
    // Step 1: Load Datasets
    // ---------------------------------------------------------
    let qa_dataset = generate_qa_dataset();
    let similarity_dataset = generate_semantic_similarity_dataset();

    println!(
        "Loaded {} Question-Answering pairs and {} Semantic Similarity pairs.\n",
        qa_dataset.len(),
        similarity_dataset.len()
    );

    // ---------------------------------------------------------
    // Step 2: Initialize BPE Tokenizer with BERT Special Tokens
    // ---------------------------------------------------------
    let special_tokens = &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

    let mut corpus = String::new();
    for sample in &qa_dataset {
        corpus.push_str(&sample.question);
        corpus.push(' ');
        corpus.push_str(&sample.context);
        corpus.push(' ');
    }
    for (s1, s2, _) in &similarity_dataset {
        corpus.push_str(s1);
        corpus.push(' ');
        corpus.push_str(s2);
        corpus.push(' ');
    }

    let tokenizer = ByteLevelBPE::train(&corpus, 320, special_tokens)?;
    let cls_id = tokenizer.special_token_id("[CLS]").unwrap_or(2);
    let sep_id = tokenizer.special_token_id("[SEP]").unwrap_or(3);
    let _pad_id = tokenizer.special_token_id("[PAD]").unwrap_or(0);

    println!(
        "Trained BPE Tokenizer (Vocab size: {}, Special tokens: {:?})\n",
        tokenizer.vocab_size(),
        special_tokens
    );

    // ---------------------------------------------------------
    // Step 3: Process QA Input Tokens, Segments, & Answer Spans
    // ---------------------------------------------------------
    struct QAItem {
        input_ids: Vec<usize>,
        token_type_ids: Vec<usize>,
        start_idx: usize,
        end_idx: usize,
    }

    let mut prepared_samples = Vec::with_capacity(qa_dataset.len());

    for sample in &qa_dataset {
        let q_tokens = tokenizer.encode(&sample.question)?;
        let c_tokens = tokenizer.encode(&sample.context)?;
        let a_tokens = tokenizer.encode(&sample.answer)?;

        // Sequence: [CLS] Question [SEP] Context [SEP]
        let mut input_ids = vec![cls_id];
        let mut token_type_ids = vec![0]; // 0 for Question segment

        input_ids.extend(&q_tokens);
        token_type_ids.extend(vec![0; q_tokens.len()]);

        input_ids.push(sep_id);
        token_type_ids.push(0);

        let context_start_pos = input_ids.len();

        input_ids.extend(&c_tokens);
        token_type_ids.extend(vec![1; c_tokens.len()]); // 1 for Context segment

        input_ids.push(sep_id);
        token_type_ids.push(1);

        // Find answer token span inside context tokens
        let mut start_idx = context_start_pos;
        let mut end_idx = context_start_pos;

        if !a_tokens.is_empty() {
            for idx in 0..=c_tokens.len().saturating_sub(a_tokens.len()) {
                if c_tokens[idx..idx + a_tokens.len()] == a_tokens[..] {
                    start_idx = context_start_pos + idx;
                    end_idx = context_start_pos + idx + a_tokens.len() - 1;
                    break;
                }
            }
        }

        prepared_samples.push(QAItem {
            input_ids,
            token_type_ids,
            start_idx,
            end_idx,
        });
    }

    let max_len = prepared_samples
        .iter()
        .map(|s| s.input_ids.len())
        .max()
        .unwrap_or(32);

    println!(
        "Prepared {} QA token sequences (Max sequence length: {})\n",
        prepared_samples.len(),
        max_len
    );

    // ---------------------------------------------------------
    // Step 4: Configure BERT Architecture
    // ---------------------------------------------------------
    let config = BertConfig {
        vocab_size: tokenizer.vocab_size(),
        d_model: 64,
        num_layers: 2,
        num_heads: 4,
        d_ff: 160,
        max_position_embeddings: 128,
        type_vocab_size: 2,
        layer_norm_eps: 1e-6,
    };

    println!("BERT Architecture Configuration:");
    println!("  • Vocabulary Size: {}", config.vocab_size);
    println!("  • Model Dimension (d_model): {}", config.d_model);
    println!("  • Transformer Encoder Layers: {}", config.num_layers);
    println!(
        "  • Self-Attention Heads (Bidirectional): {}",
        config.num_heads
    );
    println!("  • FFN Hidden Dimension: {}", config.d_ff);
    println!(
        "  • Max Position Embeddings: {}\n",
        config.max_position_embeddings
    );

    let qa_model = BertForQuestionAnswering::new(config.clone());
    let mut optimizer = Adam::new(qa_model.parameters(), 0.004);

    // ---------------------------------------------------------
    // Step 5: Train BertForQuestionAnswering
    // ---------------------------------------------------------
    let epochs = 20;
    let n_samples = prepared_samples.len();

    println!(
        "Training BertForQuestionAnswering model for {} epochs...",
        epochs
    );

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut exact_match_count = 0;

        for item in &prepared_samples {
            let cur_len = item.input_ids.len();

            let input_raw = RawTensor::from_vec(
                item.input_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            );
            let input_tensor = Tensor::new(input_raw, false);

            let type_raw = RawTensor::from_vec(
                item.token_type_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            );
            let type_tensor = Tensor::new(type_raw, false);

            let (start_logits, end_logits) =
                qa_model.forward_qa(&input_tensor, Some(&type_tensor))?;

            let loss_start =
                CrossEntropyLoss::forward_with_indices(&start_logits, &[item.start_idx])?;
            let loss_end = CrossEntropyLoss::forward_with_indices(&end_logits, &[item.end_idx])?;
            let total_sample_loss = loss_start.add(&loss_end)?;

            optimizer.zero_grad();
            total_sample_loss.backward();
            optimizer.step()?;

            total_loss += total_sample_loss.item();

            let pred_start = start_logits.data().argmax(1)?[0];
            let pred_end = end_logits.data().argmax(1)?[0];

            if pred_start == item.start_idx && pred_end == item.end_idx {
                exact_match_count += 1;
            }
        }

        let avg_loss = total_loss / (n_samples as f32);
        let em_score = (exact_match_count as f32) / (n_samples as f32) * 100.0;

        if epoch % 4 == 0 || epoch == 1 || epoch == epochs {
            println!(
                "Epoch {:2}/{} | Joint Span Loss: {:6.4} | Exact Match (EM): {:5.1}% ({}/{})",
                epoch, epochs, avg_loss, em_score, exact_match_count, n_samples
            );
        }
    }

    // ---------------------------------------------------------
    // Step 6: Extractive Question Answering Inference
    // ---------------------------------------------------------
    println!("\n------------------------------------------------------------");
    println!("             Extractive Question Answering Inference        ");
    println!("------------------------------------------------------------\n");

    let demo_queries = [0, 1, 3, 5];
    for &idx in &demo_queries {
        let sample = &qa_dataset[idx];
        let item = &prepared_samples[idx];
        let cur_len = item.input_ids.len();

        let input_tensor = Tensor::new(
            RawTensor::from_vec(
                item.input_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            ),
            false,
        );
        let type_tensor = Tensor::new(
            RawTensor::from_vec(
                item.token_type_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            ),
            false,
        );

        let (start_logits, end_logits) = qa_model.forward_qa(&input_tensor, Some(&type_tensor))?;

        let pred_start = start_logits.data().argmax(1)?[0];
        let pred_end = end_logits.data().argmax(1)?[0].max(pred_start);

        let extracted_tokens = if pred_start < cur_len && pred_end < cur_len {
            &item.input_ids[pred_start..=pred_end]
        } else {
            &item.input_ids[..]
        };

        let predicted_answer = tokenizer.decode(extracted_tokens)?;

        println!("Q: \"{}\"", sample.question);
        println!("Context: \"{}\"", sample.context);
        println!("Ground Truth Answer: \"{}\"", sample.answer);
        println!("Extracted Answer:    \"{}\"\n", predicted_answer.trim());
    }

    // ---------------------------------------------------------
    // Step 7: Semantic Text Embeddings & Cosine Similarity Search
    // ---------------------------------------------------------
    println!("------------------------------------------------------------");
    println!("         Semantic Text Embeddings & Cosine Search           ");
    println!("------------------------------------------------------------\n");

    let emb_model = BertForSequenceEmbedding {
        bert: qa_model.bert.clone(),
    };

    let query = "What features make Rust secure and fast?";
    let candidates = [
        "Rust is a systems programming language that focuses on safety and performance.",
        "The transformer architecture was introduced by Vaswani in 2017.",
        "Deep neural network optimizers update parameter weights via gradient descent.",
        "Rust provides high performance and memory safety.",
        "Apples and oranges are sweet fruits.",
    ];

    println!("Query: \"{}\"\n", query);

    let mut query_tokens = vec![cls_id];
    query_tokens.extend(tokenizer.encode(query)?);
    query_tokens.push(sep_id);

    let query_tensor = Tensor::new(
        RawTensor::from_vec(
            query_tokens.iter().map(|&t| t as f32).collect(),
            vec![1, query_tokens.len()],
        ),
        false,
    );

    let query_emb = emb_model.forward_embedding(&query_tensor, None)?;
    let query_vec = query_emb.data().to_contiguous();

    let mut scored_candidates = Vec::new();

    for &cand in &candidates {
        let mut cand_tokens = vec![cls_id];
        cand_tokens.extend(tokenizer.encode(cand)?);
        cand_tokens.push(sep_id);

        let cand_tensor = Tensor::new(
            RawTensor::from_vec(
                cand_tokens.iter().map(|&t| t as f32).collect(),
                vec![1, cand_tokens.len()],
            ),
            false,
        );

        let cand_emb = emb_model.forward_embedding(&cand_tensor, None)?;
        let cand_vec = cand_emb.data().to_contiguous();

        let similarity =
            BertForSequenceEmbedding::cosine_similarity(query_vec.as_slice(), cand_vec.as_slice());
        scored_candidates.push((cand, similarity));
    }

    // Rank candidates by descending cosine similarity
    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Ranked Candidates by Semantic Similarity:");
    for (rank, (doc, score)) in scored_candidates.iter().enumerate() {
        println!("  {:2}. [Score: {:6.4}] \"{}\"", rank + 1, score, doc);
    }

    // ---------------------------------------------------------
    // Step 8: Save Trained Model to SafeTensors
    // ---------------------------------------------------------
    let _ = std::fs::create_dir_all("target");
    let save_path = "target/bert_qa_embeddings.safetensors";
    let mut tensor_map = HashMap::new();

    tensor_map.insert(
        "embeddings.word_embeddings.weight".to_string(),
        qa_model.bert.embeddings.word_embeddings.weight.data(),
    );
    tensor_map.insert(
        "embeddings.position_embeddings.weight".to_string(),
        qa_model.bert.embeddings.position_embeddings.weight.data(),
    );
    tensor_map.insert(
        "embeddings.token_type_embeddings.weight".to_string(),
        qa_model.bert.embeddings.token_type_embeddings.weight.data(),
    );
    tensor_map.insert(
        "embeddings.layer_norm.weight".to_string(),
        qa_model.bert.embeddings.layer_norm.weight.data(),
    );
    tensor_map.insert(
        "embeddings.layer_norm.bias".to_string(),
        qa_model.bert.embeddings.layer_norm.bias.data(),
    );

    for (idx, layer) in qa_model.bert.encoder.layers.iter().enumerate() {
        tensor_map.insert(
            format!("encoder.layers.{}.attention.q_proj.weight", idx),
            layer.attention.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.attention.k_proj.weight", idx),
            layer.attention.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.attention.v_proj.weight", idx),
            layer.attention.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.attention.out_proj.weight", idx),
            layer.attention.out_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.attention_norm.weight", idx),
            layer.attention_norm.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.attention_norm.bias", idx),
            layer.attention_norm.bias.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.intermediate.weight", idx),
            layer.intermediate.weight.data(),
        );
        if let Some(ref b) = layer.intermediate.bias {
            tensor_map.insert(
                format!("encoder.layers.{}.intermediate.bias", idx),
                b.data(),
            );
        }
        tensor_map.insert(
            format!("encoder.layers.{}.output_dense.weight", idx),
            layer.output_dense.weight.data(),
        );
        if let Some(ref b) = layer.output_dense.bias {
            tensor_map.insert(
                format!("encoder.layers.{}.output_dense.bias", idx),
                b.data(),
            );
        }
        tensor_map.insert(
            format!("encoder.layers.{}.output_norm.weight", idx),
            layer.output_norm.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.layers.{}.output_norm.bias", idx),
            layer.output_norm.bias.data(),
        );
    }

    tensor_map.insert(
        "pooler.dense.weight".to_string(),
        qa_model.bert.pooler.dense.weight.data(),
    );
    if let Some(ref b) = qa_model.bert.pooler.dense.bias {
        tensor_map.insert("pooler.dense.bias".to_string(), b.data());
    }
    tensor_map.insert(
        "qa_outputs.weight".to_string(),
        qa_model.qa_outputs.weight.data(),
    );
    if let Some(ref b) = qa_model.qa_outputs.bias {
        tensor_map.insert("qa_outputs.bias".to_string(), b.data());
    }

    save_safetensors(&tensor_map, save_path)?;
    println!("\nSaved trained BERT model weights to {}", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        loaded.len()
    );

    println!("\nBERT QA and Text Embedding pipeline completed successfully!");
    Ok(())
}
