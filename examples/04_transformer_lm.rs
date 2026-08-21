//! Example 4: Training a Decoder-only Causal Transformer Language Model (nanoGPT architecture).

use neural_network_engine::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 04_transformer_lm: Causal Transformer Language Model (nanoGPT) ");
    println!("============================================================\n");

    let text = "to be or not to be, that is the question: whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles!";
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

    // 2. Hyperparameters
    let seq_len = 32;
    let d_model = 64;
    let num_heads = 4;
    let num_layers = 2;
    let epochs = 80;

    println!(
        "Transformer architecture: {} layers, {} heads, d_model={}, max_seq_len={}",
        num_layers, num_heads, d_model, seq_len
    );

    // 3. Initialize Transformer Language Model
    let model = TransformerLM::new(vocab_size, seq_len, d_model, num_heads, num_layers);
    let mut optimizer = Adam::adamw(model.parameters(), 0.005, 0.01);

    println!(
        "Model initialized with {} parameters.",
        model.parameters().len()
    );
    println!("\nStarting training loop ({} epochs)...", epochs);

    // Create chunks of (input, target) pairs of length `seq_len`
    let stride = 8;
    let mut samples: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut start = 0;
    while start + seq_len < total_tokens {
        let chunk_in = all_token_ids[start..start + seq_len].to_vec();
        let chunk_target = all_token_ids[start + 1..start + seq_len + 1].to_vec();
        samples.push((chunk_in, chunk_target));
        start += stride;
    }

    if samples.is_empty() {
        // Fallback for short texts: single chunk
        let len = (total_tokens - 1).min(seq_len);
        samples.push((
            all_token_ids[..len].to_vec(),
            all_token_ids[1..len + 1].to_vec(),
        ));
    }

    println!("Created {} training sequence windows.\n", samples.len());

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total_preds = 0;

        for (inp, targ) in &samples {
            let cur_len = inp.len();
            // Forward pass -> [1, cur_len, vocab_size]
            let logits = model.forward_tokens(inp, 1, cur_len)?;

            // Reshape logits to [cur_len, vocab_size] for CrossEntropyLoss
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

    // 4. Autoregressive Text Generation with Temperature Sampling
    println!("\n------------------------------------------------------------");
    println!("Autoregressive Transformer Text Generation:");
    println!("------------------------------------------------------------");

    let prompt = "to be or not ";
    print!("Prompt: \"{}\"\nGenerated: \"{}", prompt, prompt);

    let mut generated_tokens: Vec<usize> = prompt
        .chars()
        .filter_map(|c| char_to_id.get(&c).copied())
        .collect();

    let mut rng = rand::thread_rng();
    let temperature = 0.7f32;

    for _ in 0..80 {
        // Take up to `seq_len` last tokens as context
        let ctx_start = generated_tokens.len().saturating_sub(seq_len);
        let context = &generated_tokens[ctx_start..];
        let ctx_len = context.len();

        let logits = model.forward_tokens(context, 1, ctx_len)?;
        // Extract logits for the last token -> [1, vocab_size]
        let logits_2d = logits.reshape(&[ctx_len, vocab_size])?;
        let last_logits = logits_2d.slice(0, ctx_len - 1, ctx_len)?;

        // Apply temperature scaling
        let scaled_logits = last_logits.div_scalar(temperature)?;
        let probs = scaled_logits.softmax(1)?;
        let probs_data = probs.data();
        let prob_slice = probs_data.as_slice();

        // Sample next token from probability distribution
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
        print!("{}", next_char);
        generated_tokens.push(next_id);
    }
    println!("\"\n");

    println!("Transformer model training and generation demonstration complete!");
    Ok(())
}
