//! Example 11: End-to-End Language Model Pipeline:
//! 1. Load TinyStories text corpus.
//! 2. Train a Byte-Level Byte-Pair Encoding (BPE) Tokenizer from scratch.
//! 3. Export and reload tokenizer to/from JSON.
//! 4. Train a LLaMA 2 Language Model with Grouped-Query Attention (GQA), RoPE, RMSNorm & SwiGLU.
//! 5. Autoregressively generate new story text using temperature and top-k sampling.
//! 6. Save model weights to SafeTensors format.

use neural_network_engine::prelude::*;
use rand::Rng;
use std::collections::HashMap;

/// Samples next token index from logits with temperature scaling and top-k filtering.
fn sample_next_token(logits_slice: &[f32], temperature: f32, top_k: usize) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits_slice.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k truncation
    let k = top_k.min(indexed.len()).max(1);
    indexed.truncate(k);

    // Apply temperature and compute softmax probabilities
    let inv_temp = 1.0 / temperature.max(1e-4);
    let max_val = indexed[0].1;
    let mut exps: Vec<f32> = indexed
        .iter()
        .map(|&(_, v)| ((v - max_val) * inv_temp).exp())
        .collect();
    let sum_exp: f32 = exps.iter().sum();
    for e in exps.iter_mut() {
        *e /= sum_exp;
    }

    // Cumulative distribution sampling
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in exps.iter().enumerate() {
        cum += p;
        if r <= cum || i == exps.len() - 1 {
            return indexed[i].0;
        }
    }

    indexed[0].0
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("   11_llama_bpe_training: BPE Tokenizer + LLaMA 2 Pipeline  ");
    println!("============================================================\n");

    // ---------------------------------------------------------
    // Step 1: Load TinyStories text corpus
    // ---------------------------------------------------------
    let max_chars = Some(15_000);
    let corpus = load_tinystories_dataset(max_chars);
    println!(
        "Loaded TinyStories corpus: {} characters (~{} words)\n",
        corpus.len(),
        corpus.split_whitespace().count()
    );

    println!("--- Sample text preview ---");
    let preview: String = corpus.chars().take(200).collect();
    println!("\"{}...\"\n", preview.trim());

    // ---------------------------------------------------------
    // Step 2: Train Byte-Level BPE Tokenizer from scratch
    // ---------------------------------------------------------
    let target_vocab_size = 350;
    let special_tokens = &["<unk>", "<s>", "</s>", "<pad>"];

    println!(
        "Training Byte-Level BPE Tokenizer (target vocab: {})...",
        target_vocab_size
    );
    let tokenizer = ByteLevelBPE::train(&corpus, target_vocab_size, special_tokens)?;
    println!(
        "✓ Tokenizer trained successfully! Final vocabulary size: {}\n",
        tokenizer.vocab_size()
    );

    // ---------------------------------------------------------
    // Step 3: Export & Reload Tokenizer JSON
    // ---------------------------------------------------------
    let _ = std::fs::create_dir_all("target");
    let tokenizer_json_path = "target/tinystories_bpe.json";
    tokenizer.save_json(tokenizer_json_path)?;
    println!("Saved tokenizer JSON to {}", tokenizer_json_path);

    let reloaded_tokenizer = ByteLevelBPE::load_json(tokenizer_json_path)?;
    println!(
        "✓ Reloaded tokenizer successfully (verified vocab size: {})\n",
        reloaded_tokenizer.vocab_size()
    );

    // ---------------------------------------------------------
    // Step 4: Tokenize Corpus & Build Training Batches
    // ---------------------------------------------------------
    let all_tokens = tokenizer.encode(&corpus)?;
    let compression_ratio = (corpus.len() as f32) / (all_tokens.len() as f32);
    println!(
        "Corpus tokenized into {} tokens (Compression: {:.2}x vs raw bytes)",
        all_tokens.len(),
        compression_ratio
    );

    let seq_len = 24;
    let batch_size = 16;
    let num_sequences = (all_tokens.len() - 1) / seq_len;

    println!(
        "Preparing {} training sequences of length {} (Batch size: {})\n",
        num_sequences, seq_len, batch_size
    );

    // ---------------------------------------------------------
    // Step 5: Configure LLaMA 2 Model with GQA & RoPE
    // ---------------------------------------------------------
    let config = LlamaConfig {
        vocab_size: tokenizer.vocab_size(),
        d_model: 64,
        hidden_dim: 160,
        num_heads: 4,    // 4 Query heads
        num_kv_heads: 2, // 2 KV heads (GQA with G = 2)
        num_layers: 2,
        max_seq_len: 48,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    println!("LLaMA 2 Architecture Configuration:");
    println!("  • Vocabulary Size: {}", config.vocab_size);
    println!("  • Model Dimension (d_model): {}", config.d_model);
    println!("  • SwiGLU Hidden Dimension: {}", config.hidden_dim);
    println!("  • Query Attention Heads: {}", config.num_heads);
    println!(
        "  • Key/Value Attention Heads: {} (GQA G={})",
        config.num_kv_heads,
        config.num_heads / config.num_kv_heads
    );
    println!("  • Transformer Blocks: {}", config.num_layers);
    println!("  • Positional Embedding: Rotary Position Embeddings (RoPE)\n");

    let model = Llama2LM::new(config.clone());
    let mut optimizer = Adam::new(model.parameters(), 0.003);

    // ---------------------------------------------------------
    // Step 6: Train LLaMA 2 Model
    // ---------------------------------------------------------
    let epochs = 15;
    let num_batches = num_sequences / batch_size;

    println!("Training LLaMA 2 model for {} epochs...", epochs);

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for b in 0..num_batches {
            let start_seq = b * batch_size;
            let mut batch_x = Vec::with_capacity(batch_size * seq_len);
            let mut batch_y = Vec::with_capacity(batch_size * seq_len);

            for i in 0..batch_size {
                let seq_idx = start_seq + i;
                let offset = seq_idx * seq_len;
                batch_x.extend_from_slice(&all_tokens[offset..offset + seq_len]);
                batch_y.extend_from_slice(&all_tokens[offset + 1..offset + seq_len + 1]);
            }

            let logits = model.forward_tokens(&batch_x, batch_size, seq_len, 0)?;
            // Reshape [B * T, V]
            let logits_2d = logits.reshape(&[batch_size * seq_len, config.vocab_size])?;
            let loss = CrossEntropyLoss::forward_with_indices(&logits_2d, &batch_y)?;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()?;

            total_loss += loss.item();
            batch_count += 1;
        }

        let avg_loss = total_loss / (batch_count.max(1) as f32);
        let perplexity = avg_loss.exp();

        if epoch % 3 == 0 || epoch == 1 || epoch == epochs {
            println!(
                "Epoch {:2}/{} | Avg Loss: {:6.4} | Perplexity: {:6.2}",
                epoch, epochs, avg_loss, perplexity
            );
        }
    }

    // ---------------------------------------------------------
    // Step 7: Autoregressive Text Generation
    // ---------------------------------------------------------
    println!("\n------------------------------------------------------------");
    println!("          Autoregressive Story Generation via BPE           ");
    println!("------------------------------------------------------------\n");

    let prompts = ["<s> Once upon a time, Lily", "<s> Tim and his dog"];

    for &prompt in &prompts {
        println!("Prompt: \"{}\"", prompt);
        let mut gen_tokens = tokenizer.encode(prompt)?;

        for _ in 0..20 {
            let start = gen_tokens.len().saturating_sub(seq_len);
            let input_slice = &gen_tokens[start..];
            let cur_len = input_slice.len();

            let logits = model.forward_tokens(input_slice, 1, cur_len, 0)?;
            let slice = logits.data().to_contiguous();
            let num_classes = config.vocab_size;
            let last_token_logits =
                &slice.as_slice()[(cur_len - 1) * num_classes..cur_len * num_classes];

            let next_token = sample_next_token(last_token_logits, 0.7, 5);
            gen_tokens.push(next_token);

            if Some(next_token) == tokenizer.eos_token_id() {
                break;
            }
        }

        let generated_text = tokenizer.decode(&gen_tokens)?;
        println!("Generated:\n\"{}\"\n", generated_text.trim());
    }

    // ---------------------------------------------------------
    // Step 8: Save Trained Model to SafeTensors
    // ---------------------------------------------------------
    let model_save_path = "target/tinystories_llama.safetensors";
    let mut tensor_map = HashMap::new();
    tensor_map.insert(
        "tok_embeddings.weight".to_string(),
        model.tok_embeddings.weight.data(),
    );

    for (idx, block) in model.layers.iter().enumerate() {
        tensor_map.insert(
            format!("layers.{}.attn_norm.weight", idx),
            block.attn_norm.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.attn.q_proj.weight", idx),
            block.attn.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.attn.k_proj.weight", idx),
            block.attn.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.attn.v_proj.weight", idx),
            block.attn.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.attn.o_proj.weight", idx),
            block.attn.o_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.ffn_norm.weight", idx),
            block.ffn_norm.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.ffn.gate_proj.weight", idx),
            block.ffn.gate_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.ffn.up_proj.weight", idx),
            block.ffn.up_proj.weight.data(),
        );
        tensor_map.insert(
            format!("layers.{}.ffn.down_proj.weight", idx),
            block.ffn.down_proj.weight.data(),
        );
    }

    tensor_map.insert("norm.weight".to_string(), model.norm.weight.data());
    tensor_map.insert("lm_head.weight".to_string(), model.lm_head.weight.data());

    save_safetensors(&tensor_map, model_save_path)?;
    println!("Saved trained LLaMA 2 model weights to {}", model_save_path);

    let reloaded = load_safetensors(model_save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        reloaded.len()
    );

    println!("\nBPE Tokenizer and LLaMA 2 training pipeline completed successfully!");
    Ok(())
}
