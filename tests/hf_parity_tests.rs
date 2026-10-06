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
