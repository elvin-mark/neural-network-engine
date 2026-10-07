//! Standalone Speech-to-Text Transcription CLI using OpenAI Whisper.
//!
//! Features:
//! - Pure-Rust zero-dependency RIFF/WAV decoding (8, 16, 24, 32-bit PCM, 32-bit float)
//! - Automatic stereo-to-mono downmixing and linear resampling to 16 kHz
//! - 80-channel Log-Mel Spectrogram acoustic frontend
//! - Whisper tiny encoder-decoder speech recognition with SafeTensors weights
//! - Autoregressive decoding with repetition penalty and optional sampling
//! - Plain text, JSON, or SRT formatted output

use neural_network_engine::autograd::NoGradGuard;
use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::whisper::{Whisper, WhisperConfig, WhisperKVCache};
use neural_network_engine::tensor::RawTensor;
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::utils::audio::{
    compute_whisper_mel_spectrogram, read_wav_file, WavAudio,
};
use neural_network_engine::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::path::PathBuf;
use std::process;

#[derive(Debug)]
struct CliArgs {
    audio_path: Option<PathBuf>,
    model_dir: PathBuf,
    language: String,
    task: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    format: String,
    timestamps: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            audio_path: None,
            model_dir: PathBuf::from("checkpoints/whisper"),
            language: "en".to_string(),
            task: "transcribe".to_string(),
            max_tokens: 448,
            temperature: 0.0,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: None,
            format: "txt".to_string(),
            timestamps: false,
        }
    }
}

fn print_help() {
    println!(
        r#"Neural Network Engine - Speech-to-Text Transcription CLI (Whisper)

USAGE:
    cargo run --release --bin transcribe -- [OPTIONS] <AUDIO_FILE.wav>

ARGS:
    <AUDIO_FILE>                  Path to input .wav audio file

OPTIONS:
    -a, --audio <PATH>            Alternative option to specify the audio file
    -m, --model-dir <DIR>         Directory containing model.safetensors and tokenizer.json
                                  (alias: --checkpoint-dir, -c) [default: checkpoints/whisper]
    -l, --language <LANG>         Target language code (e.g. en, fr, de, es, zh, ja)
                                  [default: en]
        --task <TASK>             Task: transcribe or translate [default: transcribe]
    -n, --max-tokens <N>          Maximum target text tokens to generate [default: 448]
    -t, --temperature <F>         Sampling temperature: 0.0 for greedy decoding [default: 0.0]
        --top-p <F>               Nucleus sampling probability threshold [default: 0.9]
        --repetition-penalty <F>  Penalty applied to repeated tokens [default: 1.1]
    -f, --format <FMT>            Output format: txt, json, srt [default: txt]
        --timestamps              Include estimated segment timestamps in output
    -s, --seed <N>                Random seed for stochastic generation
    -h, --help                    Print help information

EXAMPLES:
    # Transcribe a WAV file with default model (checkpoints/whisper)
    cargo run --release --bin transcribe -- audio.wav

    # Transcribe specifying custom model directory
    cargo run --release --bin transcribe -- audio.wav --model-dir checkpoints/whisper

    # Transcribe with JSON output
    cargo run --release --bin transcribe -- audio.wav --format json

    # Transcribe French audio
    cargo run --release --bin transcribe -- french_audio.wav --language fr
"#
    );
}

fn parse_args() -> Result<CliArgs> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let mut args = CliArgs::default();
    let mut i = 0;

    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "-h" || arg == "--help" {
            print_help();
            process::exit(0);
        } else if arg == "-a" || arg == "--audio" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --audio".to_string(),
                ));
            }
            args.audio_path = Some(PathBuf::from(&raw_args[i]));
        } else if let Some(val) = arg.strip_prefix("--audio=") {
            args.audio_path = Some(PathBuf::from(val));
        } else if arg == "-m"
            || arg == "--model-dir"
            || arg == "--model_dir"
            || arg == "--checkpoint-dir"
            || arg == "-c"
        {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --model-dir".to_string(),
                ));
            }
            args.model_dir = PathBuf::from(&raw_args[i]);
        } else if let Some(val) = arg
            .strip_prefix("--model-dir=")
            .or_else(|| arg.strip_prefix("--model_dir="))
            .or_else(|| arg.strip_prefix("--checkpoint-dir="))
            .or_else(|| arg.strip_prefix("-m="))
            .or_else(|| arg.strip_prefix("-c="))
        {
            args.model_dir = PathBuf::from(val);
        } else if arg == "-l" || arg == "--language" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --language".to_string(),
                ));
            }
            args.language = raw_args[i].to_lowercase();
        } else if let Some(val) = arg
            .strip_prefix("--language=")
            .or_else(|| arg.strip_prefix("-l="))
        {
            args.language = val.to_lowercase();
        } else if arg == "--task" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --task".to_string(),
                ));
            }
            args.task = raw_args[i].to_lowercase();
        } else if let Some(val) = arg.strip_prefix("--task=") {
            args.task = val.to_lowercase();
        } else if arg == "-n" || arg == "--max-tokens" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --max-tokens".to_string(),
                ));
            }
            args.max_tokens = raw_args[i]
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid max-tokens".to_string()))?;
        } else if let Some(val) = arg
            .strip_prefix("--max-tokens=")
            .or_else(|| arg.strip_prefix("-n="))
        {
            args.max_tokens = val
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid max-tokens".to_string()))?;
        } else if arg == "-t" || arg == "--temperature" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --temperature".to_string(),
                ));
            }
            args.temperature = raw_args[i]
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid temperature".to_string()))?;
        } else if let Some(val) = arg
            .strip_prefix("--temperature=")
            .or_else(|| arg.strip_prefix("-t="))
        {
            args.temperature = val
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid temperature".to_string()))?;
        } else if arg == "--top-p" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --top-p".to_string(),
                ));
            }
            args.top_p = raw_args[i]
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid top-p".to_string()))?;
        } else if let Some(val) = arg.strip_prefix("--top-p=") {
            args.top_p = val
                .parse()
                .map_err(|_| EngineError::InvalidArgument("Invalid top-p".to_string()))?;
        } else if arg == "--repetition-penalty" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --repetition-penalty".to_string(),
                ));
            }
            args.repetition_penalty = raw_args[i].parse().map_err(|_| {
                EngineError::InvalidArgument("Invalid repetition-penalty".to_string())
            })?;
        } else if let Some(val) = arg.strip_prefix("--repetition-penalty=") {
            args.repetition_penalty = val.parse().map_err(|_| {
                EngineError::InvalidArgument("Invalid repetition-penalty".to_string())
            })?;
        } else if arg == "-f" || arg == "--format" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --format".to_string(),
                ));
            }
            args.format = raw_args[i].to_lowercase();
        } else if let Some(val) = arg
            .strip_prefix("--format=")
            .or_else(|| arg.strip_prefix("-f="))
        {
            args.format = val.to_lowercase();
        } else if arg == "--timestamps" {
            args.timestamps = true;
        } else if arg == "-s" || arg == "--seed" {
            i += 1;
            if i >= raw_args.len() {
                return Err(EngineError::InvalidArgument(
                    "Missing argument for --seed".to_string(),
                ));
            }
            args.seed = Some(
                raw_args[i]
                    .parse()
                    .map_err(|_| EngineError::InvalidArgument("Invalid seed".to_string()))?,
            );
        } else if let Some(val) = arg
            .strip_prefix("--seed=")
            .or_else(|| arg.strip_prefix("-s="))
        {
            args.seed = Some(
                val.parse()
                    .map_err(|_| EngineError::InvalidArgument("Invalid seed".to_string()))?,
            );
        } else if !arg.starts_with('-') {
            if args.audio_path.is_none() {
                args.audio_path = Some(PathBuf::from(arg));
            } else {
                return Err(EngineError::InvalidArgument(format!(
                    "Unexpected positional argument '{}'",
                    arg
                )));
            }
        } else {
            return Err(EngineError::InvalidArgument(format!(
                "Unknown option '{}'",
                arg
            )));
        }
        i += 1;
    }

    Ok(args)
}

/// Applies temperature, repetition penalty, and top-p nucleus sampling to logits.
fn sample_next_token(
    logits: &[f32],
    generated_tokens: &[usize],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    rng: &mut StdRng,
) -> usize {
    let vocab_size = logits.len();
    let mut modified = logits.to_vec();

    // 1. Repetition penalty
    if repetition_penalty != 1.0 {
        for &tok in generated_tokens {
            if tok < vocab_size {
                if modified[tok] < 0.0 {
                    modified[tok] *= repetition_penalty;
                } else {
                    modified[tok] /= repetition_penalty;
                }
            }
        }
    }

    // 2. Greedy selection if temperature <= 0.0
    if temperature <= 0.0 {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in modified.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        return best_idx;
    }

    // 3. Scale by temperature
    for v in &mut modified {
        *v /= temperature;
    }

    // 4. Softmax
    let max_logit = modified.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0f32;
    for v in &mut modified {
        *v = (*v - max_logit).exp();
        exp_sum += *v;
    }
    for v in &mut modified {
        *v /= exp_sum.max(1e-12);
    }

    // 5. Top-P (Nucleus) Truncation
    let mut indexed: Vec<(usize, f32)> = modified.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    indexed.truncate(cutoff);

    // Renormalize
    let filtered_sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
    let r: f32 = rng.gen::<f32>() * filtered_sum;

    let mut running = 0.0f32;
    for &(token, prob) in &indexed {
        running += prob;
        if running >= r {
            return token;
        }
    }

    indexed[0].0
}

/// Prepares the official 80-channel Log-Mel Spectrogram padded to Whisper's 30-second window (3000 frames).
fn prepare_whisper_mel(wav: &WavAudio) -> Result<Tensor> {
    let mel = compute_whisper_mel_spectrogram(&wav.samples);
    let batched = mel.unsqueeze(0)?;
    Ok(Tensor::new(batched, false))
}

/// Identifies standard Whisper special prompt tokens by text or fallback IDs.
fn get_prompt_token_ids(
    tokenizer: &HfTokenizer,
    language: &str,
    task: &str,
) -> (Vec<usize>, usize) {
    let sot = tokenizer
        .token_to_id("<|startoftranscript|>")
        .unwrap_or(50258);
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50257);

    // Language token, e.g. <|en|>
    let lang_str = format!("<|{}|>", language);
    let lang_id = tokenizer
        .token_to_id(&lang_str)
        .or_else(|| tokenizer.token_to_id("<|en|>"))
        .unwrap_or(50259);

    // Task token, e.g. <|transcribe|> or <|translate|>
    let task_str = format!("<|{}|>", task);
    let task_id = tokenizer
        .token_to_id(&task_str)
        .or_else(|| tokenizer.token_to_id("<|transcribe|>"))
        .unwrap_or(50359);

    let notimestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap_or(50363);

    let prompt_tokens = vec![sot, lang_id, task_id, notimestamps];
    (prompt_tokens, eot)
}

fn main() -> Result<()> {
    let _no_grad = NoGradGuard::new();
    let args = parse_args()?;

    let audio_path = args.audio_path.ok_or_else(|| {
        eprintln!("Error: No input audio file provided.\n");
        print_help();
        EngineError::InvalidArgument("Audio file is required".to_string())
    })?;

    if !audio_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Input audio file not found: {:?}",
            audio_path
        )));
    }

    let (model_weights_path, tokenizer_path) = if args.model_dir.is_file() {
        let parent = args
            .model_dir
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        (args.model_dir.clone(), parent.join("tokenizer.json"))
    } else {
        (
            args.model_dir.join("model.safetensors"),
            args.model_dir.join("tokenizer.json"),
        )
    };

    if !model_weights_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Whisper model weights not found at {:?}. Use -m / --model-dir to specify checkpoint directory.",
            model_weights_path
        )));
    }
    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Whisper tokenizer configuration not found at {:?}. Use -m / --model-dir to specify checkpoint directory.",
            tokenizer_path
        )));
    }

    // 1. Read & Preprocess WAV Audio
    eprintln!("[*] Loading audio from {:?}...", audio_path);
    let mut wav = read_wav_file(&audio_path)?;
    let orig_sr = wav.sample_rate;
    let orig_channels = wav.channels;
    let duration = wav.duration_seconds();

    eprintln!(
        "[*] Audio info: {:.2}s duration, {} channels, {} Hz sample rate ({} bits)",
        duration, orig_channels, orig_sr, wav.bits_per_sample
    );

    if wav.sample_rate != 16000 {
        eprintln!(
            "[*] Resampling audio from {} Hz to 16000 Hz...",
            wav.sample_rate
        );
        wav.resample(16000);
    }

    // 2. Extract Acoustic Log-Mel Spectrogram
    eprintln!("[*] Computing 80-channel Log-Mel Spectrogram...");
    let mel = prepare_whisper_mel(&wav)?;

    // 3. Load Whisper Model & Tokenizer
    eprintln!(
        "[*] Loading Whisper model weights from {:?}...",
        model_weights_path
    );
    let config = WhisperConfig::whisper_tiny();
    let mut model = Whisper::new(config);
    model.load_safetensors(&model_weights_path)?;

    eprintln!("[*] Loading Whisper tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    // 4. Encode Audio Spectrogram
    eprintln!("[*] Running Whisper Encoder...");
    let memory = model.encode(&mel)?;

    // 5. Autoregressive Transcription with Key-Value Caching
    let (prompt_tokens, eot_id) = get_prompt_token_ids(&tokenizer, &args.language, &args.task);
    let prompt_len = prompt_tokens.len();
    let mut generated = prompt_tokens.clone();

    let mut rng = match args.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    eprintln!(
        "[*] Decoding speech (task={}, lang={}, max_tokens={})...",
        args.task, args.language, args.max_tokens
    );

    let mut kv_cache = WhisperKVCache::new(model.config.decoder_layers);

    // Prompt prefill
    let prompt_raw = RawTensor::from_slice(
        &prompt_tokens.iter().map(|&t| t as f32).collect::<Vec<_>>(),
        &[1, prompt_len],
    );
    let prompt_tensor = Tensor::new(prompt_raw, false);
    let logits = model.decode_cached(&prompt_tensor, &memory, 0, Some(&mut kv_cache))?;
    let raw = logits.data().to_contiguous();
    let num_classes = model.config.vocab_size;
    let last_logits = &raw.as_slice()[(prompt_len - 1) * num_classes..prompt_len * num_classes];

    let mut next_token = sample_next_token(
        last_logits,
        &[],
        args.temperature,
        args.top_p,
        args.repetition_penalty,
        &mut rng,
    );

    for step in 0..args.max_tokens {
        if next_token == eot_id {
            break;
        }

        generated.push(next_token);

        if step + 1 < args.max_tokens {
            let cur_pos = prompt_len + step;
            let token_raw = RawTensor::from_vec(vec![next_token as f32], vec![1, 1]);
            let token_tensor = Tensor::new(token_raw, false);

            let logits =
                model.decode_cached(&token_tensor, &memory, cur_pos, Some(&mut kv_cache))?;
            let raw = logits.data().to_contiguous();
            let s = raw.as_slice();
            let last_logits = &s[(s.len() - num_classes)..];

            next_token = sample_next_token(
                last_logits,
                &generated[prompt_len..],
                args.temperature,
                args.top_p,
                args.repetition_penalty,
                &mut rng,
            );
        }
    }

    // 6. Decode Tokens to String
    let text_tokens = &generated[prompt_len..];
    let full_transcription = tokenizer.decode(text_tokens);
    let cleaned_text = full_transcription.trim().to_string();

    // 7. Render Output According to Requested Format
    match args.format.as_str() {
        "json" => {
            let json_output = serde_json::json!({
                "file": audio_path.to_string_lossy(),
                "duration_seconds": duration,
                "sample_rate": orig_sr,
                "channels": orig_channels,
                "language": args.language,
                "task": args.task,
                "text": cleaned_text,
                "tokens": text_tokens,
            });
            println!("{}", serde_json::to_string_pretty(&json_output).unwrap());
        }
        "srt" => {
            println!("1");
            let mins = (duration / 60.0).floor() as u32;
            let secs = (duration % 60.0).floor() as u32;
            let millis = ((duration % 1.0) * 1000.0).round() as u32;
            println!(
                "00:00:00,000 --> {:02}:{:02}:{:02},{:03}",
                mins / 60,
                mins % 60,
                secs,
                millis
            );
            println!("{}\n", cleaned_text);
        }
        _ => {
            if args.timestamps {
                println!("[00:00.000 --> {:05.2}] {}", duration, cleaned_text);
            } else {
                println!("{}", cleaned_text);
            }
        }
    }

    Ok(())
}
