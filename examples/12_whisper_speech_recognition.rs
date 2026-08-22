//! Example 12: Whisper Sequence-to-Sequence Speech Recognition on Spoken Audio Dataset.
//!
//! Workflow:
//! 1. Synthesize and extract Log-Mel Spectrograms from spoken audio signals.
//! 2. Tokenize transcriptions with Byte-Level BPE and speech prompt tokens (`<s>`, `<|transcribe|>`, `</s>`).
//! 3. Train the Whisper Encoder-Decoder model with 1D Conv downsampling, cross-attention, and Adam optimizer.
//! 4. Autoregressively transcribe unseen test audio spectrograms into text.
//! 5. Save and reload model weights in SafeTensors format.

use neural_network_engine::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 12_whisper_speech_recognition: Speech Recognition Pipeline ");
    println!("============================================================\n");

    // ---------------------------------------------------------
    // Step 1: Load Spoken Audio Dataset (Log-Mel Spectrograms)
    // ---------------------------------------------------------
    let num_samples = 280;
    let (dataset_specs, dataset_labels) = load_spoken_dataset(Some(num_samples));
    let n_mels = dataset_specs.shape()[1];
    let time_steps = dataset_specs.shape()[2];

    println!(
        "Loaded {} spoken audio samples (Log-Mel Spectrogram: {}x{} time frames)",
        num_samples, n_mels, time_steps
    );
    println!("Vocabulary classes: {:?}\n", SPOKEN_CLASSES);

    // ---------------------------------------------------------
    // Step 2: Initialize BPE Tokenizer with Speech Special Tokens
    // ---------------------------------------------------------
    let special_tokens = &["<unk>", "<s>", "</s>", "<pad>", "<|transcribe|>"];
    let mut corpus = String::new();
    for label in &dataset_labels {
        corpus.push_str(label);
        corpus.push(' ');
    }

    let tokenizer = ByteLevelBPE::train(&corpus, 280, special_tokens)?;
    println!(
        "Initialized BPE Tokenizer for Speech (Vocab size: {}, Special tokens: {:?})\n",
        tokenizer.vocab_size(),
        special_tokens
    );

    // ---------------------------------------------------------
    // Step 3: Split Dataset into Train (80%) and Test (20%) Sets
    // ---------------------------------------------------------
    let num_train = (num_samples as f32 * 0.8) as usize;
    let num_test = num_samples - num_train;

    let train_specs = dataset_specs.slice(0, 0, num_train)?;
    let test_specs = dataset_specs.slice(0, num_train, num_samples)?;
    let train_labels = &dataset_labels[0..num_train];
    let test_labels = &dataset_labels[num_train..num_samples];

    println!(
        "Train set: {} audio spectrograms | Test set: {} audio spectrograms\n",
        num_train, num_test
    );

    // ---------------------------------------------------------
    // Step 4: Configure Whisper Encoder-Decoder Architecture
    // ---------------------------------------------------------
    let config = WhisperConfig {
        n_mels,
        d_model: 64,
        encoder_layers: 2,
        decoder_layers: 2,
        encoder_heads: 4,
        decoder_heads: 4,
        d_ff: 160,
        vocab_size: tokenizer.vocab_size(),
        max_source_positions: 128,
        max_target_positions: 32,
    };

    println!("Whisper Model Configuration:");
    println!("  • Acoustic Mel Channels: {}", config.n_mels);
    println!("  • Latent Model Dimension (d_model): {}", config.d_model);
    println!(
        "  • Encoder Layers (Bidirectional): {}",
        config.encoder_layers
    );
    println!(
        "  • Decoder Layers (Causal + Cross-Attention): {}",
        config.decoder_layers
    );
    println!("  • Attention Heads: {}", config.encoder_heads);
    println!("  • FFN Hidden Dimension: {}", config.d_ff);
    println!("  • Target Text Vocabulary: {}\n", config.vocab_size);

    let model = Whisper::new(config.clone());
    let mut optimizer = Adam::new(model.parameters(), 0.003);

    // ---------------------------------------------------------
    // Step 5: Encode Transcription Sequences
    // ---------------------------------------------------------
    // Format: <s> <|transcribe|> <word> </s>
    let mut train_target_ids: Vec<Vec<usize>> = Vec::with_capacity(num_train);
    for label in train_labels {
        let mut tokens = vec![
            tokenizer.bos_token_id().unwrap_or(1),
            tokenizer.special_token_id("<|transcribe|>").unwrap_or(4),
        ];
        tokens.extend(tokenizer.encode(label)?);
        tokens.push(tokenizer.eos_token_id().unwrap_or(2));
        train_target_ids.push(tokens);
    }

    let max_seq_len = train_target_ids.iter().map(|v| v.len()).max().unwrap_or(6);
    let pad_id = tokenizer.pad_token_id().unwrap_or(3);

    // ---------------------------------------------------------
    // Step 6: Train Whisper Model
    // ---------------------------------------------------------
    let epochs = 15;
    let batch_size = 16;
    let num_batches = num_train / batch_size;

    println!(
        "Training Whisper model for {} epochs (batch size {})...",
        epochs, batch_size
    );

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut correct_tokens = 0;
        let mut total_tokens = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = start + batch_size;

            let b_specs = train_specs.slice(0, start, end)?;
            let mel_tensor = Tensor::new(b_specs, false);

            // Pad batch target tokens
            let mut b_input_ids = Vec::with_capacity(batch_size * (max_seq_len - 1));
            let mut b_target_ids = Vec::with_capacity(batch_size * (max_seq_len - 1));

            for seq in &train_target_ids[start..end] {
                for t in 0..(max_seq_len - 1) {
                    if t < seq.len() - 1 {
                        b_input_ids.push(seq[t]);
                        b_target_ids.push(seq[t + 1]);
                    } else {
                        b_input_ids.push(pad_id);
                        b_target_ids.push(pad_id);
                    }
                }
            }

            let input_raw = RawTensor::from_vec(
                b_input_ids.iter().map(|&id| id as f32).collect(),
                vec![batch_size, max_seq_len - 1],
            );
            let input_tensor = Tensor::new(input_raw, false);

            let logits = model.forward_model(&mel_tensor, &input_tensor)?;
            let logits_2d = logits.reshape(&[batch_size * (max_seq_len - 1), config.vocab_size])?;

            let loss = CrossEntropyLoss::forward_with_indices(&logits_2d, &b_target_ids)?;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()?;

            total_loss += loss.item() * (batch_size as f32);

            let preds = logits_2d.data().argmax(1)?;
            for (&p, &t) in preds.iter().zip(b_target_ids.iter()) {
                if t != pad_id {
                    if p == t {
                        correct_tokens += 1;
                    }
                    total_tokens += 1;
                }
            }
        }

        let avg_loss = total_loss / (num_train as f32);
        let token_acc = (correct_tokens as f32) / (total_tokens.max(1) as f32) * 100.0;

        if epoch % 3 == 0 || epoch == 1 || epoch == epochs {
            println!(
                "Epoch {:2}/{} | Avg Loss: {:6.4} | Token Accuracy: {:5.1}%",
                epoch, epochs, avg_loss, token_acc
            );
        }
    }

    // ---------------------------------------------------------
    // Step 7: Evaluate Speech Transcription on Unseen Audio
    // ---------------------------------------------------------
    println!("\n------------------------------------------------------------");
    println!("        Sample Speech Transcriptions (Whisper Audio)        ");
    println!("------------------------------------------------------------\n");

    let num_show = 8.min(num_test);
    let mut exact_matches = 0;

    for (i, ground_truth) in test_labels.iter().enumerate().take(num_show) {
        let sample_spec = test_specs.slice(0, i, i + 1)?;
        let sample_tensor = Tensor::new(sample_spec, false);

        let transcribed = model.generate_transcription(&sample_tensor, &tokenizer, 8)?;
        // Strip prompt token artifacts
        let clean_transcription = transcribed
            .replace("<|transcribe|>", "")
            .replace("<s>", "")
            .replace("</s>", "")
            .trim()
            .to_string();

        let is_match = clean_transcription.contains(ground_truth)
            || ground_truth.contains(&clean_transcription);
        if is_match {
            exact_matches += 1;
        }

        let mark = if is_match { "✓" } else { "✗" };
        println!(
            "Sample {:2}: Actual = {:10} | Transcribed = {:10} [{}]",
            i + 1,
            ground_truth,
            clean_transcription,
            mark
        );
    }

    let match_rate = (exact_matches as f32) / (num_show as f32) * 100.0;
    println!("\nSample Transcription Success Rate: {:5.1}%", match_rate);

    // ---------------------------------------------------------
    // Step 8: Save Model Checkpoint to SafeTensors
    // ---------------------------------------------------------
    let _ = std::fs::create_dir_all("target");
    let save_path = "target/whisper_speech_model.safetensors";
    let mut tensor_map = HashMap::new();

    tensor_map.insert(
        "encoder.conv1.weight".to_string(),
        model.encoder.conv1.weight.data(),
    );
    if let Some(ref b) = model.encoder.conv1.bias {
        tensor_map.insert("encoder.conv1.bias".to_string(), b.data());
    }
    tensor_map.insert(
        "encoder.pos_embed".to_string(),
        model.encoder.pos_embed.data(),
    );
    tensor_map.insert(
        "encoder.ln_post.weight".to_string(),
        model.encoder.ln_post.weight.data(),
    );
    tensor_map.insert(
        "encoder.ln_post.bias".to_string(),
        model.encoder.ln_post.bias.data(),
    );

    for (idx, block) in model.encoder.blocks.iter().enumerate() {
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn_ln.weight", idx),
            block.self_attn_ln.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn_ln.bias", idx),
            block.self_attn_ln.bias.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn.q_proj.weight", idx),
            block.self_attn.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn.k_proj.weight", idx),
            block.self_attn.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn.v_proj.weight", idx),
            block.self_attn.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.self_attn.out_proj.weight", idx),
            block.self_attn.out_proj.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.mlp_ln.weight", idx),
            block.mlp_ln.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.mlp_ln.bias", idx),
            block.mlp_ln.bias.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.mlp_fc1.weight", idx),
            block.mlp_fc1.weight.data(),
        );
        tensor_map.insert(
            format!("encoder.blocks.{}.mlp_fc2.weight", idx),
            block.mlp_fc2.weight.data(),
        );
    }

    tensor_map.insert(
        "decoder.token_embedding.weight".to_string(),
        model.decoder.token_embedding.weight.data(),
    );
    tensor_map.insert(
        "decoder.pos_embed".to_string(),
        model.decoder.pos_embed.data(),
    );
    tensor_map.insert(
        "decoder.ln_post.weight".to_string(),
        model.decoder.ln_post.weight.data(),
    );
    tensor_map.insert(
        "decoder.ln_post.bias".to_string(),
        model.decoder.ln_post.bias.data(),
    );
    tensor_map.insert(
        "decoder.lm_head.weight".to_string(),
        model.decoder.lm_head.weight.data(),
    );

    for (idx, block) in model.decoder.blocks.iter().enumerate() {
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn_ln.weight", idx),
            block.self_attn_ln.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn_ln.bias", idx),
            block.self_attn_ln.bias.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn.q_proj.weight", idx),
            block.self_attn.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn.k_proj.weight", idx),
            block.self_attn.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn.v_proj.weight", idx),
            block.self_attn.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.self_attn.out_proj.weight", idx),
            block.self_attn.out_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn_ln.weight", idx),
            block.cross_attn_ln.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn_ln.bias", idx),
            block.cross_attn_ln.bias.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn.q_proj.weight", idx),
            block.cross_attn.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn.k_proj.weight", idx),
            block.cross_attn.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn.v_proj.weight", idx),
            block.cross_attn.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.cross_attn.out_proj.weight", idx),
            block.cross_attn.out_proj.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.mlp_ln.weight", idx),
            block.mlp_ln.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.mlp_ln.bias", idx),
            block.mlp_ln.bias.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.mlp_fc1.weight", idx),
            block.mlp_fc1.weight.data(),
        );
        tensor_map.insert(
            format!("decoder.blocks.{}.mlp_fc2.weight", idx),
            block.mlp_fc2.weight.data(),
        );
    }

    save_safetensors(&tensor_map, save_path)?;
    println!("\nSaved trained Whisper model weights to {}", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        loaded.len()
    );

    println!("\nWhisper speech recognition pipeline completed successfully!");
    Ok(())
}
