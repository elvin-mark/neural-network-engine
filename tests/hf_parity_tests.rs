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
        &ref_data.input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );
    let token_type_ids = Tensor::from_slice(
        &ref_data.token_type_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
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

    println!("Expected first 5 seq tokens: {:?}", ref_data.first_5_seq_output);
    println!("Actual first 5 seq tokens:   {:?}", actual_first_5_seq);
    println!("Expected first 5 pooler:     {:?}", ref_data.first_5_pooled_output);
    println!("Actual first 5 pooler:       {:?}", actual_first_5_pooled);

    for (a, b) in actual_first_5_seq.iter().zip(ref_data.first_5_seq_output.iter()) {
        assert!(
            (a - b).abs() < 2e-3,
            "Sequence representation mismatch: {} vs {}",
            a,
            b
        );
    }

    for (a, b) in actual_first_5_pooled.iter().zip(ref_data.first_5_pooled_output.iter()) {
        assert!(
            (a - b).abs() < 2e-3,
            "Pooled representation mismatch: {} vs {}",
            a,
            b
        );
    }
}
