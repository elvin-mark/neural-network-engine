//! Example 5: Training a LLaMA 2 Model with Grouped-Query Attention (GQA),
//! Rotary Position Embeddings (RoPE), RMSNorm, and SwiGLU FFN.

use neural_network_engine::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 05_llama2_gqa: LLaMA 2 Architecture with GQA, RoPE & SwiGLU ");
    println!("============================================================\n");

    let text = "knowledge is power, and with great knowledge comes great responsibility. we explore the universe through curiosity and reason.";
    println!("Training text:\n\"{}\"\n", text);

    // 1. Build character vocabulary
    let mut unique_chars: Vec<char> = text.chars().collect::<HashSet<_>>().into_iter().collect();
    unique_chars.sort();

    let vocab_size = unique_chars.len();
    let char_to_id: HashMap<char, usize> = unique_chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let id_to_char: HashMap<usize, char> = unique_chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, c))
        .collect();

    println!("Vocabulary: {} unique characters", vocab_size);

    let all_token_ids: Vec<usize> = text.chars().map(|c| char_to_id[&c]).collect();
    let total_tokens = all_token_ids.len();

    // 2. Hyperparameters & LLaMA 2 Config with Grouped-Query Attention (GQA)
    let seq_len = 32;
    let config = LlamaConfig {
        vocab_size,
        d_model: 64,
        hidden_dim: 160,
        num_heads: 4,    // 4 Query heads
        num_kv_heads: 2, // 2 KV heads (GQA with G = 4 / 2 = 2 query heads per KV head)
        num_layers: 2,
        max_seq_len: 48,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    println!("LLaMA 2 Model Configuration:");
    println!("  - Layers: {}", config.num_layers);
    println!("  - d_model: {}", config.d_model);
    println!("  - Hidden dim (SwiGLU): {}", config.hidden_dim);
    println!("  - Query heads (N_q): {}", config.num_heads);
    println!(
        "  - Key/Value heads (N_kv): {} (GQA G=2)",
        config.num_kv_heads
    );
    println!("  - Normalization: RMSNorm (eps={})", config.norm_eps);
    println!(
        "  - Positional Embedding: Rotary (RoPE theta={})",
        config.rope_theta
    );

    // 3. Initialize Model & AdamW Optimizer
    let model = Llama2LM::new(config.clone());
    let mut optimizer = Adam::adamw(model.parameters(), 0.005, 0.01);

    println!(
        "Model initialized with {} parameter tensors.\n",
        model.parameters().len()
    );

    // 4. Create sequence training windows
    let stride = 6;
    let mut samples: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut start = 0;
    while start + seq_len < total_tokens {
        let chunk_in = all_token_ids[start..start + seq_len].to_vec();
        let chunk_target = all_token_ids[start + 1..start + seq_len + 1].to_vec();
        samples.push((chunk_in, chunk_target));
        start += stride;
    }

    if samples.is_empty() {
        let len = (total_tokens - 1).min(seq_len);
        samples.push((
            all_token_ids[..len].to_vec(),
            all_token_ids[1..len + 1].to_vec(),
        ));
    }

    println!("Created {} training sequence windows.\n", samples.len());
    let epochs = 80;
    println!("Starting training loop ({} epochs)...", epochs);

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total_preds = 0;

        for (inp, targ) in &samples {
            let cur_len = inp.len();
            // Forward pass with RoPE starting at position 0
            let logits = model.forward_tokens(inp, 1, cur_len, 0)?;

            let logits_2d = logits.reshape(&[cur_len, vocab_size])?;
            let loss = CrossEntropyLoss::forward_with_indices(&logits_2d, targ)?;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()?;

            total_loss += loss.item();

            let preds = logits_2d.data().argmax(1)?;
            for (&p, &t) in preds.iter().zip(targ.iter()) {
                if p == t {
                    correct += 1;
                }
                total_preds += 1;
            }
        }

        let avg_loss = total_loss / (samples.len() as f32);
        let accuracy = (correct as f32) / (total_preds as f32) * 100.0;

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:2}/{} | Loss: {:7.4} | Token Accuracy: {:6.2}% ({}/{})",
                epoch, epochs, avg_loss, accuracy, correct, total_preds
            );
        }
    }

    // 5. Autoregressive Text Generation with RoPE and GQA
    println!("\n------------------------------------------------------------");
    println!("Autoregressive LLaMA 2 Text Generation:");
    println!("------------------------------------------------------------");

    let prompt = "knowledge is ";
    print!("Prompt: \"{prompt}\"\nGenerated: \"{prompt}");

    let mut generated_tokens: Vec<usize> = prompt
        .chars()
        .filter_map(|c| char_to_id.get(&c).copied())
        .collect();

    let mut rng = rand::thread_rng();
    let temperature = 0.7f32;

    for _ in 0..60 {
        let ctx_start = generated_tokens.len().saturating_sub(seq_len);
        let context = &generated_tokens[ctx_start..];
        let ctx_len = context.len();

        let logits = model.forward_tokens(context, 1, ctx_len, 0)?;
        let logits_2d = logits.reshape(&[ctx_len, vocab_size])?;
        let last_logits = logits_2d.slice(0, ctx_len - 1, ctx_len)?;

        let scaled_logits = last_logits.div_scalar(temperature)?;
        let probs = scaled_logits.softmax(1)?;
        let probs_data = probs.data();
        let prob_slice = probs_data.as_slice();

        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let mut next_id = 0;
        for (i, &p) in prob_slice.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                next_id = i;
                break;
            }
        }

        let next_char = id_to_char.get(&next_id).copied().unwrap_or('?');
        print!("{next_char}");
        generated_tokens.push(next_id);
    }
    println!("\"\n");

    println!("LLaMA 2 model training and text generation demonstration complete!");
    Ok(())
}
