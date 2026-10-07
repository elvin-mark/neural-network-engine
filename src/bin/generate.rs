//! Standalone Text Generation & Inference CLI for Neural Network Engine.
//!
//! Supports autoregressive generation, interactive REPL, and extractive QA
//! using converted checkpoints (GPT-2, TinyLlamas, Dynamic TinyBERT, Whisper).

use neural_network_engine::autograd::NoGradGuard;
use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::bert::{BertConfig, BertForQuestionAnswering, BertModel};
use neural_network_engine::models::gpt2::{GPT2Config, GPT2Model};
use neural_network_engine::models::llama::{Llama2LM, LlamaConfig};
use neural_network_engine::models::whisper::{Whisper, WhisperConfig};
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::Tensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CliArgs {
    model: String,
    checkpoint_dir: Option<PathBuf>,
    prompt: Option<String>,
    context: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model: "gpt2".to_string(),
            checkpoint_dir: None,
            prompt: None,
            context: None,
            max_tokens: 50,
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            seed: None,
        }
    }
}

fn print_help() {
    println!(
        r#"Neural Network Engine - Text Generation & Inference CLI

USAGE:
    cargo run --bin generate -- [OPTIONS]

OPTIONS:
    -m, --model <MODEL>           Model architecture to run:
                                  gpt2, tinyllamas, tinybert, minilm, whisper
                                  [default: gpt2]
    -c, --checkpoint-dir <DIR>    Path to checkpoint directory containing model.safetensors
                                  and tokenizer.json [default: checkpoints/<model>]
    -p, --prompt <TEXT>           Input prompt text. If omitted, starts interactive REPL
        --context <TEXT>          Context passage for extractive QA (used by tinybert)
    -n, --max-tokens <N>          Maximum new tokens to generate [default: 50]
    -t, --temperature <FLOAT>     Sampling temperature (<= 0.0 for greedy) [default: 0.7]
        --top-k <INT>             Top-k candidate tokens filtering (0 to disable) [default: 40]
        --top-p <FLOAT>           Top-p nucleus cumulative probability (1.0 to disable) [default: 0.9]
    -s, --seed <INT>              Seed for pseudo-random sampling [optional]
    -h, --help                    Print help information

EXAMPLES:
    # 1. Deterministic greedy generation with GPT-2:
    cargo run --bin generate -- --model gpt2 --prompt "The quick brown fox" --temperature 0.0

    # 2. Creative generation with TinyLlamas (TinyStories 15M):
    cargo run --bin generate -- --model tinyllamas --prompt "Once upon a time" --temperature 0.8 --top-p 0.9

    # 3. Question Answering with Dynamic TinyBERT:
    cargo run --bin generate -- --model tinybert --context "Paris is the capital of France." --prompt "Where is the capital of France?"

    # 4. Interactive REPL:
    cargo run --bin generate -- --model gpt2
"#
    );
}

fn parse_args() -> std::result::Result<CliArgs, String> {
    let mut args = CliArgs::default();
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;

    while i < raw_args.len() {
        let arg = &raw_args[i];

        if arg == "-h" || arg == "--help" {
            print_help();
            std::process::exit(0);
        } else if arg == "-m" || arg == "--model" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --model".to_string());
            }
            args.model = raw_args[i].to_lowercase();
        } else if let Some(val) = arg.strip_prefix("--model=") {
            args.model = val.to_lowercase();
        } else if arg == "-c" || arg == "--checkpoint-dir" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --checkpoint-dir".to_string());
            }
            args.checkpoint_dir = Some(PathBuf::from(&raw_args[i]));
        } else if let Some(val) = arg.strip_prefix("--checkpoint-dir=") {
            args.checkpoint_dir = Some(PathBuf::from(val));
        } else if arg == "-p" || arg == "--prompt" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --prompt".to_string());
            }
            args.prompt = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--prompt=") {
            args.prompt = Some(val.to_string());
        } else if arg == "--context" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --context".to_string());
            }
            args.context = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--context=") {
            args.context = Some(val.to_string());
        } else if arg == "-n" || arg == "--max-tokens" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --max-tokens".to_string());
            }
            args.max_tokens = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid max-tokens: {}", e))?;
        } else if let Some(val) = arg.strip_prefix("--max-tokens=") {
            args.max_tokens = val
                .parse()
                .map_err(|e| format!("Invalid max-tokens: {}", e))?;
        } else if arg == "-t" || arg == "--temperature" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --temperature".to_string());
            }
            args.temperature = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid temperature: {}", e))?;
        } else if let Some(val) = arg.strip_prefix("--temperature=") {
            args.temperature = val
                .parse()
                .map_err(|e| format!("Invalid temperature: {}", e))?;
        } else if arg == "--top-k" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --top-k".to_string());
            }
            let k: usize = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid top-k: {}", e))?;
            args.top_k = if k > 0 { Some(k) } else { None };
        } else if let Some(val) = arg.strip_prefix("--top-k=") {
            let k: usize = val.parse().map_err(|e| format!("Invalid top-k: {}", e))?;
            args.top_k = if k > 0 { Some(k) } else { None };
        } else if arg == "--top-p" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --top-p".to_string());
            }
            let p: f32 = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid top-p: {}", e))?;
            args.top_p = if p < 1.0 { Some(p) } else { None };
        } else if let Some(val) = arg.strip_prefix("--top-p=") {
            let p: f32 = val.parse().map_err(|e| format!("Invalid top-p: {}", e))?;
            args.top_p = if p < 1.0 { Some(p) } else { None };
        } else if arg == "-s" || arg == "--seed" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --seed".to_string());
            }
            args.seed = Some(
                raw_args[i]
                    .parse()
                    .map_err(|e| format!("Invalid seed: {}", e))?,
            );
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            args.seed = Some(val.parse().map_err(|e| format!("Invalid seed: {}", e))?);
        } else {
            return Err(format!("Unknown option: {}", arg));
        }

        i += 1;
    }

    Ok(args)
}

fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

fn run_gpt2(args: CliArgs, checkpoint_dir: &Path) -> Result<()> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    if !model_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Model file not found at {:?}",
            model_path
        )));
    }
    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Tokenizer file not found at {:?}",
            tokenizer_path
        )));
    }

    eprintln!("Loading GPT-2 tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading GPT-2 model weights from {:?}...", model_path);
    let config = GPT2Config::gpt2_small(tokenizer.vocab_size());
    let mut model = GPT2Model::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    let mut rng = create_rng(args.seed);

    if let Some(prompt) = args.prompt {
        let prompt_tokens = tokenizer.encode(&prompt);
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidArgument("Prompt is empty".to_string()));
        }

        print!("{}", prompt);
        io::stdout().flush()?;

        model.generate_stream(
            &tokenizer,
            &prompt_tokens,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            tokenizer.eos_token_id(),
            &mut rng,
            |_id, piece| {
                print!("{}", piece);
                io::stdout().flush()?;
                Ok(true)
            },
        )?;
        println!();
    } else {
        println!("==============================================================");
        println!("Neural Network Engine - Interactive REPL (GPT-2 124M)");
        println!(
            "Config: temp={}, top-k={:?}, top-p={:?}, max-tokens={}",
            args.temperature, args.top_k, args.top_p, args.max_tokens
        );
        println!("Type '/quit' or '/exit' to exit.");
        println!("==============================================================\n");

        let stdin = io::stdin();
        let mut reader = stdin.lock();

        loop {
            print!(">>> ");
            io::stdout().flush()?;

            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "/quit" || trimmed == "/exit" {
                println!("Goodbye!");
                break;
            }

            let prompt_tokens = tokenizer.encode(trimmed);
            if prompt_tokens.is_empty() {
                continue;
            }

            print!("{}", trimmed);
            io::stdout().flush()?;

            model.generate_stream(
                &tokenizer,
                &prompt_tokens,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                tokenizer.eos_token_id(),
                &mut rng,
                |_id, piece| {
                    print!("{}", piece);
                    io::stdout().flush()?;
                    Ok(true)
                },
            )?;
            println!("\n");
        }
    }

    Ok(())
}

fn run_tinyllamas(args: CliArgs, checkpoint_dir: &Path) -> Result<()> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    if !model_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Model file not found at {:?}",
            model_path
        )));
    }
    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Tokenizer file not found at {:?}",
            tokenizer_path
        )));
    }

    eprintln!("Loading TinyLlamas tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!(
        "Loading TinyLlamas (TinyStories 15M) weights from {:?}...",
        model_path
    );
    let config = LlamaConfig::stories15m();
    let mut model = Llama2LM::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    let mut rng = create_rng(args.seed);

    if let Some(prompt) = args.prompt {
        let prompt_tokens = tokenizer.encode(&prompt);
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidArgument("Prompt is empty".to_string()));
        }

        print!("{}", prompt);
        io::stdout().flush()?;

        model.generate_stream(
            &tokenizer,
            &prompt_tokens,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            tokenizer.eos_token_id(),
            &mut rng,
            |_id, piece| {
                print!("{}", piece);
                io::stdout().flush()?;
                Ok(true)
            },
        )?;
        println!();
    } else {
        println!("==============================================================");
        println!("Neural Network Engine - Interactive REPL (TinyLlamas 15M)");
        println!(
            "Config: temp={}, top-k={:?}, top-p={:?}, max-tokens={}",
            args.temperature, args.top_k, args.top_p, args.max_tokens
        );
        println!("Type '/quit' or '/exit' to exit.");
        println!("==============================================================\n");

        let stdin = io::stdin();
        let mut reader = stdin.lock();

        loop {
            print!(">>> ");
            io::stdout().flush()?;

            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "/quit" || trimmed == "/exit" {
                println!("Goodbye!");
                break;
            }

            let prompt_tokens = tokenizer.encode(trimmed);
            if prompt_tokens.is_empty() {
                continue;
            }

            print!("{}", trimmed);
            io::stdout().flush()?;

            model.generate_stream(
                &tokenizer,
                &prompt_tokens,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                tokenizer.eos_token_id(),
                &mut rng,
                |_id, piece| {
                    print!("{}", piece);
                    io::stdout().flush()?;
                    Ok(true)
                },
            )?;
            println!("\n");
        }
    }

    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn answer_question(
    model: &BertForQuestionAnswering,
    tokenizer: &HfTokenizer,
    context: &str,
    question: &str,
) -> Result<String> {
    let cls_id = tokenizer.cls_token_id().unwrap_or(101);
    let sep_id = tokenizer.sep_token_id().unwrap_or(102);

    let q_tokens = tokenizer.encode(question);
    let c_tokens = tokenizer.encode(context);

    // Format: [CLS] question [SEP] context [SEP]
    let mut input_ids = Vec::with_capacity(q_tokens.len() + c_tokens.len() + 3);
    let mut token_type_ids = Vec::with_capacity(input_ids.capacity());

    input_ids.push(cls_id);
    token_type_ids.push(0);

    for &tok in &q_tokens {
        input_ids.push(tok);
        token_type_ids.push(0);
    }

    input_ids.push(sep_id);
    token_type_ids.push(0);

    let context_start_idx = input_ids.len();
    for &tok in &c_tokens {
        input_ids.push(tok);
        token_type_ids.push(1);
    }

    let context_end_idx = input_ids.len(); // exclusive
    input_ids.push(sep_id);
    token_type_ids.push(1);

    let seq_len = input_ids.len();
    let input_tensor = Tensor::from_slice(
        &input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );
    let type_tensor = Tensor::from_slice(
        &token_type_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );

    let (start_logits, end_logits) = model.forward_qa(&input_tensor, Some(&type_tensor))?;

    let start_slice = start_logits.data().to_contiguous();
    let end_slice = end_logits.data().to_contiguous();

    let start_s = start_slice.as_slice();
    let end_s = end_slice.as_slice();

    // Find best span (s, e) within context where s <= e
    let mut best_score = f32::NEG_INFINITY;
    let mut best_start = context_start_idx;
    let mut best_end = context_start_idx;

    for s in context_start_idx..context_end_idx {
        for e in s..std::cmp::min(s + 30, context_end_idx) {
            let score = start_s[s] + end_s[e];
            if score > best_score {
                best_score = score;
                best_start = s;
                best_end = e;
            }
        }
    }

    let answer_ids = &input_ids[best_start..=best_end];
    let answer = tokenizer.decode(answer_ids);
    Ok(answer.trim().to_string())
}

fn run_tinybert(args: CliArgs, checkpoint_dir: &Path) -> Result<()> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    if !model_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Model file not found at {:?}",
            model_path
        )));
    }
    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Tokenizer file not found at {:?}",
            tokenizer_path
        )));
    }

    eprintln!("Loading TinyBERT tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading Dynamic TinyBERT QA model from {:?}...", model_path);
    let config = BertConfig::dynamic_tinybert();
    let mut model = BertForQuestionAnswering::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    if let (Some(context), Some(question)) = (args.context.as_deref(), args.prompt.as_deref()) {
        println!("Context:  {}", context);
        println!("Question: {}", question);
        let answer = answer_question(&model, &tokenizer, context, question)?;
        println!("Answer:   {}", answer);
    } else {
        println!("==============================================================");
        println!("Neural Network Engine - Dynamic TinyBERT Question Answering");
        println!("Type '/quit' or '/exit' to exit.");
        println!("==============================================================\n");

        let stdin = io::stdin();
        let mut reader = stdin.lock();

        let mut current_context = args.context.unwrap_or_default();

        loop {
            if current_context.is_empty() {
                print!("Enter context passage:\n>>> ");
                io::stdout().flush()?;
                let mut line = String::new();
                if reader.read_line(&mut line)? == 0 {
                    break;
                }
                let trimmed = line.trim();
                if trimmed == "/quit" || trimmed == "/exit" {
                    break;
                }
                current_context = trimmed.to_string();
                if current_context.is_empty() {
                    continue;
                }
            }

            print!("Enter question (or /clear to change context, /quit to exit):\n>>> ");
            io::stdout().flush()?;

            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed == "/quit" || trimmed == "/exit" {
                break;
            }
            if trimmed == "/clear" {
                current_context.clear();
                continue;
            }
            if trimmed.is_empty() {
                continue;
            }

            let answer = answer_question(&model, &tokenizer, &current_context, trimmed)?;
            println!("Answer: {}\n", answer);
        }
    }

    Ok(())
}

fn run_minilm(args: CliArgs, checkpoint_dir: &Path) -> Result<()> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    eprintln!("Loading MiniLM tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading all-MiniLM-L6-v2 model from {:?}...", model_path);
    let config = BertConfig::all_minilm_l6_v2();
    let mut model = BertModel::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    let text = args
        .prompt
        .as_deref()
        .unwrap_or("Hello world! Deep learning in Rust.");
    let tokens = tokenizer.encode(text);
    let cls_id = tokenizer.cls_token_id().unwrap_or(101);
    let sep_id = tokenizer.sep_token_id().unwrap_or(102);

    let mut input_ids = vec![cls_id];
    input_ids.extend(&tokens);
    input_ids.push(sep_id);

    let seq_len = input_ids.len();
    let input_tensor = Tensor::from_slice(
        &input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[1, seq_len],
        false,
    );
    let type_tensor = Tensor::zeros(&[1, seq_len], false);

    let (_seq_emb, pooled_emb) = model.forward_bert(&input_tensor, Some(&type_tensor))?;
    let pooled_contig = pooled_emb.data().to_contiguous();
    let pooled_slice = pooled_contig.as_slice();

    println!("Input Text: \"{}\"", text);
    println!("Embedding Dimension: {}", pooled_slice.len());
    println!("First 5 pooled features: {:?}", &pooled_slice[..5]);
    Ok(())
}

fn run_whisper(args: CliArgs, checkpoint_dir: &Path) -> Result<()> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    eprintln!("Loading Whisper tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading Whisper Tiny model from {:?}...", model_path);
    let config = WhisperConfig::whisper_tiny();
    let mut model = Whisper::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    println!("Whisper Tiny Audio Speech-to-Text Model loaded.");
    println!("Vocabulary size: {}", tokenizer.vocab_size());

    if let Some(prompt) = args.prompt {
        println!("Prompt text: \"{}\"", prompt);
        let tokens = tokenizer.encode(&prompt);
        println!("Encoded prompt tokens: {:?}", tokens);
        println!("Decoded back: \"{}\"", tokenizer.decode(&tokens));
    } else {
        println!(
            "Whisper speech transcription requires 80-channel log-mel spectrogram audio input."
        );
        println!("Use `--prompt <text>` to test Whisper tokenizer BPE encoding and decoding.");
    }

    Ok(())
}

fn main() -> Result<()> {
    let _no_grad = NoGradGuard::new();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Use --help for usage information.");
            std::process::exit(1);
        }
    };

    let default_dir = format!("checkpoints/{}", args.model);
    let checkpoint_dir = args
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(&default_dir));

    match args.model.as_str() {
        "gpt2" => run_gpt2(args, &checkpoint_dir)?,
        "tinyllamas" | "llama" => run_tinyllamas(args, &checkpoint_dir)?,
        "tinybert" | "bert-qa" => run_tinybert(args, &checkpoint_dir)?,
        "minilm" | "bert" => run_minilm(args, &checkpoint_dir)?,
        "whisper" => run_whisper(args, &checkpoint_dir)?,
        unknown => {
            eprintln!(
                "Error: Unknown model '{}'. Supported models: gpt2, tinyllamas, tinybert, minilm, whisper",
                unknown
            );
            std::process::exit(1);
        }
    }

    Ok(())
}
