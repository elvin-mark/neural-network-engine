//! Example 3: Autoregressive Character-Level Neural Language Model with Embeddings and AdamW.

use neural_network_engine::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

struct CharLM {
    embedding: Embedding,
    ln: LayerNorm,
    fc1: Linear,
    gelu: GELU,
    fc2: Linear,
}

impl CharLM {
    pub fn new(vocab_size: usize, emb_dim: usize, hidden_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, emb_dim),
            ln: LayerNorm::new(emb_dim),
            fc1: Linear::new(emb_dim, hidden_dim),
            gelu: GELU,
            fc2: Linear::new(hidden_dim, vocab_size),
        }
    }

    pub fn forward_tokens(&self, token_indices: &[usize]) -> Result<Tensor> {
        let emb = self.embedding.forward_indices(token_indices)?;
        let norm = self.ln.forward(&emb)?;
        let h = self.fc1.forward(&norm)?;
        let act = self.gelu.forward(&h)?;
        self.fc2.forward(&act)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.embedding.parameters());
        p.extend(self.ln.parameters());
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("  03_character_lm: Autoregressive Character Language Model  ");
    println!("============================================================\n");

    let text = "rust is blazingly fast, memory safe, and concurrent. neural networks from scratch!";
    println!("Training corpus: \"{}\"\n", text);

    // Build vocabulary
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

    println!("Vocabulary size: {} unique characters.", vocab_size);

    let token_ids: Vec<usize> = text.chars().map(|c| char_to_id[&c]).collect();
    let n_tokens = token_ids.len();

    // Input characters (0..N-1) -> Target next characters (1..N)
    let inputs = &token_ids[0..n_tokens - 1];
    let targets = &token_ids[1..n_tokens];

    let model = CharLM::new(vocab_size, 32, 64);
    let mut optimizer = Adam::adamw(model.parameters(), 0.02, 0.01);

    println!("\nTraining Character LM (150 epochs with AdamW)...");
    for epoch in 1..=150 {
        let logits = model.forward_tokens(inputs)?;
        let loss = CrossEntropyLoss::forward_with_indices(&logits, targets)?;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step()?;

        if epoch % 25 == 0 || epoch == 1 {
            let preds = logits.data().argmax(1)?;
            let correct = preds
                .iter()
                .zip(targets.iter())
                .filter(|(&p, &t)| p == t)
                .count();
            let accuracy = (correct as f32) / (targets.len() as f32) * 100.0;

            println!(
                "Epoch {:3}/150 | Loss: {:7.4} | Next-Char Accuracy: {:6.2}%",
                epoch,
                loss.item(),
                accuracy
            );
        }
    }

    // Autoregressive text generation
    println!("\nGenerating text continuation (temperature sampling):");
    let prompt = "rust is ";
    print!("Prompt: \"{}\" -> Generated: \"{}", prompt, prompt);

    let mut current_char = prompt.chars().last().unwrap();
    let mut rng = rand::thread_rng();

    for _ in 0..40 {
        let current_id = char_to_id.get(&current_char).copied().unwrap_or(0);
        let logits = model.forward_tokens(&[current_id])?;
        let probs = logits.data().softmax(1)?;
        let prob_slice = probs.as_slice();

        // Sample from distribution
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
        current_char = next_char;
    }
    println!("\"\n");

    println!("Character LM demo completed successfully!");
    Ok(())
}
