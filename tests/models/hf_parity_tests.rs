use neural_network_engine::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(serde::Deserialize)]
struct ReferenceData {
    input_ids: Vec<usize>,
    generated_ids: Vec<usize>,
    generated_text: String,
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_gpt2_safetensors_and_generation_parity() {
    let checkpoint_dir = Path::new("checkpoints/gpt2");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_gpt2_safetensors_and_generation_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: ReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate standard GPT-2 model (124M parameters)
    let config = GPT2Config::gpt2_small(50257);
    let mut model = GPT2Model::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into GPT2Model");

    // 4. Run greedy autoregressive generation for 5 new tokens
    let new_tokens_to_generate = ref_data.generated_ids.len() - ref_data.input_ids.len();
    let generated = model
        .generate(&ref_data.input_ids, new_tokens_to_generate, None)
        .expect("Generation failed");

    println!("Expected generated IDs: {:?}", ref_data.generated_ids);
    println!("Actual generated IDs:   {:?}", generated);
    println!("Expected generated text: {:?}", ref_data.generated_text);

    // 5. Assert 100% exact parity with Hugging Face PyTorch generation
    assert_eq!(
        generated, ref_data.generated_ids,
        "Generated tokens deviate from Hugging Face PyTorch baseline!"
    );
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_tinyllamas_safetensors_and_generation_parity() {
    let checkpoint_dir = Path::new("checkpoints/tinyllamas");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_tinyllamas_safetensors_and_generation_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: ReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate TinyStories 15M LLaMA model
    let config = LlamaConfig::stories15m();
    let mut model = Llama2LM::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into Llama2LM");

    // 4. Run greedy autoregressive generation (temperature = 0.0) for 5 new tokens
    let new_tokens_to_generate = ref_data.generated_ids.len() - ref_data.input_ids.len();
    let generated = model
        .generate_cached(&ref_data.input_ids, new_tokens_to_generate, 0.0)
        .expect("Generation failed");

    println!("Expected generated IDs: {:?}", ref_data.generated_ids);
    println!("Actual generated IDs:   {:?}", generated);
    println!("Expected generated text: {:?}", ref_data.generated_text);

    // 5. Assert 100% exact parity with Hugging Face PyTorch generation
    assert_eq!(
        generated, ref_data.generated_ids,
        "Generated tokens deviate from Hugging Face PyTorch baseline!"
    );
}

#[derive(serde::Deserialize)]
struct BertReferenceData {
    input_ids: Vec<usize>,
    token_type_ids: Vec<usize>,
    first_5_seq_output: Vec<f32>,
    first_5_pooled_output: Vec<f32>,
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_minilm_safetensors_and_embeddings_parity() {
    let checkpoint_dir = Path::new("checkpoints/minilm");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_minilm_safetensors_and_embeddings_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: BertReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate all-MiniLM-L6-v2 BERT model
    let config = BertConfig::all_minilm_l6_v2();
    let mut model = BertModel::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into BertModel");

    // 4. Run forward pass with input_ids and token_type_ids
    let seq_len = ref_data.input_ids.len();
    let input_ids = Tensor::from_slice(
        &ref_data
            .input_ids
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );
    let token_type_ids = Tensor::from_slice(
        &ref_data
            .token_type_ids
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );

    let (seq_out, pooled_out) = model
        .forward_bert(&input_ids, Some(&token_type_ids))
        .expect("BertModel forward failed");

    // 5. Compare sequence embeddings and pooler output against Hugging Face PyTorch
    let seq_slice = seq_out.data().to_contiguous();
    let actual_first_5_seq = &seq_slice.as_slice()[..5];

    let pooled_slice = pooled_out.data().to_contiguous();
    let actual_first_5_pooled = &pooled_slice.as_slice()[..5];

    println!(
        "Expected first 5 seq tokens: {:?}",
        ref_data.first_5_seq_output
    );
    println!("Actual first 5 seq tokens:   {:?}", actual_first_5_seq);
    println!(
        "Expected first 5 pooler:     {:?}",
        ref_data.first_5_pooled_output
    );
    println!("Actual first 5 pooler:       {:?}", actual_first_5_pooled);

    for (a, b) in actual_first_5_seq
        .iter()
        .zip(ref_data.first_5_seq_output.iter())
    {
        assert!(
            (a - b).abs() < 2e-3,
            "Sequence representation mismatch: {} vs {}",
            a,
            b
        );
    }

    for (a, b) in actual_first_5_pooled
        .iter()
        .zip(ref_data.first_5_pooled_output.iter())
    {
        assert!(
            (a - b).abs() < 2e-3,
            "Pooled representation mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[derive(serde::Deserialize)]
struct QAReferenceData {
    input_ids: Vec<usize>,
    token_type_ids: Vec<usize>,
    best_start: usize,
    best_end: usize,
    first_5_start_logits: Vec<f32>,
    first_5_end_logits: Vec<f32>,
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_tinybert_safetensors_and_qa_parity() {
    let checkpoint_dir = Path::new("checkpoints/tinybert");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_tinybert_safetensors_and_qa_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: QAReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate Dynamic TinyBERT QA model
    let config = BertConfig::dynamic_tinybert();
    let mut model = BertForQuestionAnswering::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into BertForQuestionAnswering");

    // 4. Run QA forward pass
    let seq_len = ref_data.input_ids.len();
    let input_ids = Tensor::from_slice(
        &ref_data
            .input_ids
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );
    let token_type_ids = Tensor::from_slice(
        &ref_data
            .token_type_ids
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );

    let (start_logits, end_logits) = model
        .forward_qa(&input_ids, Some(&token_type_ids))
        .expect("BertForQuestionAnswering forward failed");

    let start_slice = start_logits.data().to_contiguous();
    let end_slice = end_logits.data().to_contiguous();

    let actual_start_slice = start_slice.as_slice();
    let actual_end_slice = end_slice.as_slice();

    let actual_best_start = actual_start_slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let actual_best_end = actual_end_slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!(
        "Expected best start/end: ({}, {})",
        ref_data.best_start, ref_data.best_end
    );
    println!(
        "Actual best start/end:   ({}, {})",
        actual_best_start, actual_best_end
    );

    // 5. Assert exact span prediction parity
    assert_eq!(
        actual_best_start, ref_data.best_start,
        "Best start token index mismatch!"
    );
    assert_eq!(
        actual_best_end, ref_data.best_end,
        "Best end token index mismatch!"
    );

    for (a, b) in actual_start_slice[..5]
        .iter()
        .zip(ref_data.first_5_start_logits.iter())
    {
        assert!((a - b).abs() < 5e-2, "Start logit mismatch: {} vs {}", a, b);
    }

    for (a, b) in actual_end_slice[..5]
        .iter()
        .zip(ref_data.first_5_end_logits.iter())
    {
        assert!((a - b).abs() < 5e-2, "End logit mismatch: {} vs {}", a, b);
    }
}

#[derive(serde::Deserialize)]
struct WhisperReferenceData {
    decoder_input_ids: Vec<usize>,
    next_predicted_token: usize,
    first_5_logits: Vec<f32>,
    mel_shape: Vec<usize>,
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_whisper_tiny_safetensors_and_logits_parity() {
    let checkpoint_dir = Path::new("checkpoints/whisper");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_whisper_tiny_safetensors_and_logits_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: WhisperReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate official Whisper Tiny model
    let config = WhisperConfig::whisper_tiny();
    let mut model = Whisper::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into Whisper");

    // 4. Construct input mel tensor [1, 80, 3000] (zeros)
    let b = ref_data.mel_shape[0];
    let n_mels = ref_data.mel_shape[1];
    let t_audio = ref_data.mel_shape[2];
    let mel = Tensor::zeros(&[b, n_mels, t_audio], false);

    // Construct decoder prompt tokens [1, T_tokens]
    let seq_len = ref_data.decoder_input_ids.len();
    let tokens_raw = RawTensor::from_slice(
        &ref_data
            .decoder_input_ids
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>(),
        &[1, seq_len],
    );
    let tokens = Tensor::new(tokens_raw, false);

    // 5. Run forward model
    let logits = model
        .forward_model(&mel, &tokens)
        .expect("Whisper forward_model failed");

    assert_eq!(logits.shape(), &[1, seq_len, 51865]);

    // Extract logits at the last token position
    let slice = logits.data().to_contiguous();
    let num_classes = 51865;
    let last_token_logits = &slice.as_slice()[(seq_len - 1) * num_classes..seq_len * num_classes];

    let actual_next_token = last_token_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!("Expected next token: {}", ref_data.next_predicted_token);
    println!("Actual next token:   {}", actual_next_token);
    println!("Expected first 5 logits: {:?}", ref_data.first_5_logits);
    println!("Actual first 5 logits:   {:?}", &last_token_logits[..5]);

    // 6. Assert exact token prediction parity
    assert_eq!(
        actual_next_token, ref_data.next_predicted_token,
        "Next predicted token mismatch!"
    );

    for (a, b) in last_token_logits[..5]
        .iter()
        .zip(ref_data.first_5_logits.iter())
    {
        assert!((a - b).abs() < 5e-2, "Logit mismatch: {} vs {}", a, b);
    }
}

#[derive(serde::Deserialize)]
struct ViTReferenceData {
    input_shape: Vec<usize>,
    predicted_class: usize,
    first_5_logits: Vec<f32>,
}

#[test]
#[ignore = "requires pre-downloaded HF checkpoints (local verification only)"]
fn test_hf_vit_tiny_safetensors_and_classification_parity() {
    let checkpoint_dir = Path::new("checkpoints/vit");
    let model_path = checkpoint_dir.join("model.safetensors");
    let ref_path = checkpoint_dir.join("reference.json");

    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping test_hf_vit_tiny_safetensors_and_classification_parity: checkpoint files not found at {:?}",
            checkpoint_dir
        );
        return;
    }

    // 1. Load reference test vectors
    let ref_file = File::open(&ref_path).expect("Failed to open reference.json");
    let ref_data: ViTReferenceData =
        serde_json::from_reader(BufReader::new(ref_file)).expect("Failed to parse reference.json");

    // 2. Instantiate ViT Tiny model (192 embedding dim, 12 layers, 3 heads)
    let config = ViTConfig::vit_tiny_patch16_224();
    let mut model = VisionTransformer::new(config);

    // 3. Load converted SafeTensors weights
    model
        .load_safetensors(&model_path)
        .expect("Failed to load SafeTensors weights into VisionTransformer");

    // 4. Construct input tensor [1, 3, 224, 224] (zeros)
    let shape = &ref_data.input_shape;
    let x = Tensor::zeros(&[shape[0], shape[1], shape[2], shape[3]], false);

    // 5. Forward classification
    let logits = model.forward(&x).expect("VisionTransformer forward failed");
    assert_eq!(logits.shape(), &[1, 1000]);

    let slice = logits.data().to_contiguous();
    let actual_slice = slice.as_slice();

    let actual_predicted_class = actual_slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!("Expected predicted class: {}", ref_data.predicted_class);
    println!("Actual predicted class:   {}", actual_predicted_class);
    println!("Expected first 5 logits:  {:?}", ref_data.first_5_logits);
    println!("Actual first 5 logits:    {:?}", &actual_slice[..5]);

    // 6. Assert exact classification parity
    assert_eq!(
        actual_predicted_class, ref_data.predicted_class,
        "Predicted class mismatch!"
    );

    for (a, b) in actual_slice[..5].iter().zip(ref_data.first_5_logits.iter()) {
        assert!((a - b).abs() < 5e-2, "Logit mismatch: {} vs {}", a, b);
    }
}
