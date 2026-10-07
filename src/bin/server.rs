//! High-Performance Zero-Dependency OpenAI-Compatible HTTP Server CLI for Neural Network Engine.
//!
//! Provides a drop-in local inference API for text generation, embeddings, and speech transcription:
//! - `GET  /health`                   -> Server status, uptime, and loaded models
//! - `GET  /v1/models`                -> OpenAI-compatible model catalog
//! - `POST /v1/completions`           -> Text generation (GPT-2, TinyLlama) with SSE streaming support
//! - `POST /v1/chat/completions`      -> Chat completion endpoint with prompt formatting
//! - `POST /v1/embeddings`            -> Dense embeddings (MiniLM, ModernBERT)
//! - `POST /v1/audio/transcriptions`  -> Speech-to-text audio transcription (Whisper)

use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::bert::{BertConfig, BertModel};
use neural_network_engine::models::gpt2::{GPT2Config, GPT2Model};
use neural_network_engine::models::llama::{Llama2LM, LlamaConfig};
use neural_network_engine::models::modern_bert::{ModernBertConfig, ModernBertModel};
use neural_network_engine::models::whisper::{Whisper, WhisperConfig};
use neural_network_engine::tensor::RawTensor;
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::utils::audio::{compute_whisper_mel_spectrogram, parse_wav_bytes};
use neural_network_engine::Tensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// CLI Configuration & Arguments
// ============================================================================

#[derive(Debug, Clone)]
struct ServerConfig {
    host: String,
    port: u16,
    threads: usize,
    checkpoints_dir: PathBuf,
    preload: Vec<String>,
    api_key: Option<String>,
    cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            checkpoints_dir: PathBuf::from("checkpoints"),
            preload: Vec::new(),
            api_key: None,
            cors: true,
        }
    }
}

const HELP_TEXT: &str = r#"Neural Network Engine - High-Performance OpenAI-Compatible HTTP Server CLI

USAGE:
    cargo run --release --bin server -- [OPTIONS]

OPTIONS:
    -h, --host <HOST>             Host address to bind [default: 127.0.0.1]
    -p, --port <PORT>             Port number to listen on [default: 8080]
    -t, --threads <N>             Worker thread pool capacity [default: num_cpus]
    -c, --checkpoints-dir <DIR>   Root directory for model checkpoints [default: checkpoints]
        --preload <MODELS>        Comma-separated list of models to eagerly load at startup
                                  (e.g. gpt2,whisper,minilm or "all")
        --api-key <KEY>           Optional Bearer token required for API authentication
        --no-cors                 Disable permissive CORS headers
        --help                    Print help information

SUPPORTED ENDPOINTS:
    GET  /health                  System health check and loaded model diagnostics
    GET  /v1/models               List available model architectures and loaded status
    POST /v1/completions          OpenAI-compatible text generation with optional SSE streaming
    POST /v1/chat/completions     OpenAI-compatible chat completions
    POST /v1/embeddings           OpenAI-compatible vector embeddings (MiniLM, ModernBERT)
    POST /v1/audio/transcriptions OpenAI-compatible speech-to-text (Whisper)

EXAMPLES:
    # 1. Start server with default settings on port 8080:
    cargo run --release --bin server

    # 2. Start server binding all interfaces on port 8000 with preloaded Whisper & MiniLM:
    cargo run --release --bin server -- --host 0.0.0.0 --port 8000 --preload whisper,minilm

    # 3. Test text generation with curl (non-streaming):
    curl http://127.0.0.1:8080/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt2", "prompt": "Artificial intelligence is", "max_tokens": 30}'

    # 4. Test text generation with Server-Sent Events (SSE) streaming:
    curl -N http://127.0.0.1:8080/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt2", "prompt": "Once upon a time", "stream": true}'

    # 5. Test embedding extraction with curl:
    curl http://127.0.0.1:8080/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{"model": "minilm", "input": "Deep learning in pure Rust"}'

    # 6. Test speech-to-text transcription with multipart audio file:
    curl http://127.0.0.1:8080/v1/audio/transcriptions \
        -F file=@audio.wav \
        -F model=whisper
"#;

fn print_help() {
    print!("{}", HELP_TEXT);
}

fn parse_args() -> std::result::Result<ServerConfig, String> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let mut config = ServerConfig::default();
    let mut i = 0;

    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "-h" || arg == "--help" {
            print_help();
            process::exit(0);
        } else if arg == "--host" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --host".to_string());
            }
            config.host = raw_args[i].clone();
        } else if let Some(val) = arg.strip_prefix("--host=") {
            config.host = val.to_string();
        } else if arg == "-p" || arg == "--port" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --port".to_string());
            }
            config.port = raw_args[i]
                .parse()
                .map_err(|_| "Invalid port number".to_string())?;
        } else if let Some(val) = arg
            .strip_prefix("--port=")
            .or_else(|| arg.strip_prefix("-p="))
        {
            config.port = val.parse().map_err(|_| "Invalid port number".to_string())?;
        } else if arg == "-t" || arg == "--threads" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --threads".to_string());
            }
            config.threads = raw_args[i]
                .parse()
                .map_err(|_| "Invalid thread count".to_string())?;
        } else if let Some(val) = arg
            .strip_prefix("--threads=")
            .or_else(|| arg.strip_prefix("-t="))
        {
            config.threads = val
                .parse()
                .map_err(|_| "Invalid thread count".to_string())?;
        } else if arg == "-c" || arg == "--checkpoints-dir" || arg == "--checkpoint-dir" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --checkpoints-dir".to_string());
            }
            config.checkpoints_dir = PathBuf::from(&raw_args[i]);
        } else if let Some(val) = arg
            .strip_prefix("--checkpoints-dir=")
            .or_else(|| arg.strip_prefix("--checkpoint-dir="))
            .or_else(|| arg.strip_prefix("-c="))
        {
            config.checkpoints_dir = PathBuf::from(val);
        } else if arg == "--preload" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --preload".to_string());
            }
            config.preload = raw_args[i]
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .collect();
        } else if let Some(val) = arg.strip_prefix("--preload=") {
            config.preload = val.split(',').map(|s| s.trim().to_lowercase()).collect();
        } else if arg == "--api-key" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --api-key".to_string());
            }
            config.api_key = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--api-key=") {
            config.api_key = Some(val.to_string());
        } else if arg == "--no-cors" {
            config.cors = false;
        } else {
            return Err(format!("Unknown option '{}'", arg));
        }
        i += 1;
    }

    Ok(config)
}

// ============================================================================
// Worker Thread Pool
// ============================================================================

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    sender: Option<Sender<Job>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);

        for _ in 0..size {
            let rx: Arc<Mutex<Receiver<Job>>> = Arc::clone(&receiver);
            let handle = thread::spawn(move || loop {
                let job = {
                    let lock = rx.lock().unwrap();
                    lock.recv()
                };
                match job {
                    Ok(f) => f(),
                    Err(_) => break,
                }
            });
            workers.push(handle);
        }

        Self {
            sender: Some(sender),
            workers,
        }
    }

    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(Box::new(f));
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// ============================================================================
// HTTP/1.1 Engine & Multipart / Base64 Parsing
// ============================================================================

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn parse(mut stream: &mut TcpStream) -> std::io::Result<Self> {
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Connection closed",
            ));
        }

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid HTTP request line",
            ));
        }
        let method = parts[0].to_uppercase();
        let full_path = parts[1].to_string();
        let path = full_path.split('?').next().unwrap_or("/").to_string();

        let mut headers = HashMap::new();
        let mut content_length = 0usize;

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(pos) = trimmed.find(':') {
                let key = trimmed[..pos].trim().to_lowercase();
                let val = trimmed[pos + 1..].trim().to_string();
                if key == "content-length" {
                    content_length = val.parse().unwrap_or(0);
                }
                headers.insert(key, val);
            }
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body)?;
        }

        Ok(HttpRequest {
            method,
            path,
            headers,
            body,
        })
    }

    fn json_body<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        serde_json::from_slice(&self.body)
            .map_err(|e| EngineError::InvalidArgument(format!("Malformed JSON payload: {}", e)))
    }
}

/// Helper to decode standard and URL-safe Base64 strings into raw bytes.
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut table = [255u8; 256];
    for (i, &b) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .iter()
        .enumerate()
    {
        table[b as usize] = i as u8;
    }
    table[b'-' as usize] = 62;
    table[b'_' as usize] = 63;

    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;

    for &b in &clean {
        if b == b'=' {
            break;
        }
        let val = table[b as usize];
        if val == 255 {
            return None;
        }
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Parsed Multipart/form-data container.
#[derive(Default)]
struct MultipartForm {
    fields: HashMap<String, String>,
    file_bytes: Option<Vec<u8>>,
    file_name: Option<String>,
}

fn parse_multipart(content_type: &str, body: &[u8]) -> Option<MultipartForm> {
    let boundary_marker = "boundary=";
    let boundary_idx = content_type.find(boundary_marker)?;
    let raw_boundary = &content_type[boundary_idx + boundary_marker.len()..];
    let boundary_str = raw_boundary.trim_matches('"').split(';').next()?.trim();
    if boundary_str.is_empty() {
        return None;
    }

    let delimiter = format!("--{}", boundary_str).into_bytes();
    let mut form = MultipartForm::default();
    let mut pos = 0;

    while pos < body.len() {
        // Find next boundary
        let part_start = match body[pos..]
            .windows(delimiter.len())
            .position(|w| w == delimiter.as_slice())
        {
            Some(p) => pos + p + delimiter.len(),
            None => break,
        };

        // Check if terminating boundary
        if part_start + 2 <= body.len() && &body[part_start..part_start + 2] == b"--" {
            break;
        }

        // Find end of this part (the next boundary)
        let part_end = match body[part_start..]
            .windows(delimiter.len())
            .position(|w| w == delimiter.as_slice())
        {
            Some(p) => part_start + p,
            None => body.len(),
        };

        let part_slice = &body[part_start..part_end];
        let header_delim = b"\r\n\r\n";
        if let Some(header_end) = part_slice
            .windows(header_delim.len())
            .position(|w| w == header_delim)
        {
            let header_bytes = &part_slice[..header_end];
            let content_bytes = &part_slice[header_end + 4..];
            // Trim trailing \r\n from content
            let content_bytes = if content_bytes.ends_with(b"\r\n") {
                &content_bytes[..content_bytes.len() - 2]
            } else {
                content_bytes
            };

            let header_str = String::from_utf8_lossy(header_bytes);
            let mut field_name = None;
            let mut filename = None;

            for line in header_str.lines() {
                if line.to_lowercase().starts_with("content-disposition:") {
                    for part in line.split(';') {
                        let trimmed = part.trim();
                        if let Some(name_str) = trimmed.strip_prefix("name=") {
                            field_name = Some(name_str.trim_matches('"').to_string());
                        } else if let Some(file_str) = trimmed.strip_prefix("filename=") {
                            filename = Some(file_str.trim_matches('"').to_string());
                        }
                    }
                }
            }

            if let Some(fname) = field_name {
                if fname == "file" || filename.is_some() {
                    form.file_bytes = Some(content_bytes.to_vec());
                    form.file_name = filename;
                } else {
                    form.fields.insert(
                        fname,
                        String::from_utf8_lossy(content_bytes).trim().to_string(),
                    );
                }
            }
        }

        pos = part_end;
    }

    Some(form)
}

fn send_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    body: &[u8],
    cors: bool,
) -> std::io::Result<()> {
    let mut header_buf = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_code,
        status_text,
        content_type,
        body.len()
    );
    if cors {
        header_buf.push_str("Access-Control-Allow-Origin: *\r\n");
        header_buf.push_str("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        header_buf.push_str("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    }
    header_buf.push_str("\r\n");

    stream.write_all(header_buf.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn send_json<T: Serialize>(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    payload: &T,
    cors: bool,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    send_response(
        stream,
        status_code,
        status_text,
        "application/json; charset=utf-8",
        &body,
        cors,
    )
}

fn send_error(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    message: &str,
    cors: bool,
) -> std::io::Result<()> {
    let err_obj = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": status_code
        }
    });
    send_json(stream, status_code, status_text, &err_obj, cors)
}

fn send_sse_headers(stream: &mut TcpStream, cors: bool) -> std::io::Result<()> {
    let mut headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n".to_string();
    if cors {
        headers.push_str("Access-Control-Allow-Origin: *\r\n");
        headers.push_str("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        headers.push_str("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    }
    headers.push_str("\r\n");
    stream.write_all(headers.as_bytes())?;
    stream.flush()
}

fn send_sse_chunk(stream: &mut TcpStream, event_data: &str) -> std::io::Result<()> {
    let chunk_payload = format!("data: {}\n\n", event_data);
    let chunk = format!("{:x}\r\n{}\r\n", chunk_payload.len(), chunk_payload);
    stream.write_all(chunk.as_bytes())?;
    stream.flush()
}

fn send_sse_done(stream: &mut TcpStream) -> std::io::Result<()> {
    let done_payload = "data: [DONE]\n\n";
    let chunk = format!("{:x}\r\n{}\r\n", done_payload.len(), done_payload);
    stream.write_all(chunk.as_bytes())?;
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

// ============================================================================
// Model Lifecycle & Thread-Safe Registry
// ============================================================================

struct ModelRegistry {
    checkpoints_dir: PathBuf,
    gpt2: Mutex<Option<(GPT2Model, HfTokenizer)>>,
    tinyllamas: Mutex<Option<(Llama2LM, HfTokenizer)>>,
    minilm: Mutex<Option<(BertModel, HfTokenizer)>>,
    modernbert: Mutex<Option<(ModernBertModel, HfTokenizer)>>,
    whisper: Mutex<Option<(Whisper, HfTokenizer)>>,
}

impl ModelRegistry {
    fn new(checkpoints_dir: PathBuf) -> Self {
        Self {
            checkpoints_dir,
            gpt2: Mutex::new(None),
            tinyllamas: Mutex::new(None),
            minilm: Mutex::new(None),
            modernbert: Mutex::new(None),
            whisper: Mutex::new(None),
        }
    }

    fn get_gpt2(&self) -> Result<std::sync::MutexGuard<'_, Option<(GPT2Model, HfTokenizer)>>> {
        let mut lock = self.gpt2.lock().unwrap();
        if lock.is_none() {
            let dir = self.checkpoints_dir.join("gpt2");
            let model_path = dir.join("model.safetensors");
            let tok_path = dir.join("tokenizer.json");
            if !model_path.exists() || !tok_path.exists() {
                return Err(EngineError::InvalidArgument(format!(
                    "GPT-2 checkpoint not found in {:?}. Please run conversion script first.",
                    dir
                )));
            }
            eprintln!("[*] Loading GPT-2 model from {:?}...", dir);
            let tokenizer = HfTokenizer::from_file(&tok_path)?;
            let config = GPT2Config::gpt2_small(tokenizer.vocab_size());
            let mut model = GPT2Model::new(config);
            model.load_safetensors(&model_path)?;
            *lock = Some((model, tokenizer));
        }
        Ok(lock)
    }

    fn get_tinyllamas(&self) -> Result<std::sync::MutexGuard<'_, Option<(Llama2LM, HfTokenizer)>>> {
        let mut lock = self.tinyllamas.lock().unwrap();
        if lock.is_none() {
            let dir = self.checkpoints_dir.join("tinyllamas");
            let model_path = dir.join("model.safetensors");
            let tok_path = dir.join("tokenizer.json");
            if !model_path.exists() || !tok_path.exists() {
                return Err(EngineError::InvalidArgument(format!(
                    "TinyLlamas checkpoint not found in {:?}.",
                    dir
                )));
            }
            eprintln!("[*] Loading TinyLlamas model from {:?}...", dir);
            let tokenizer = HfTokenizer::from_file(&tok_path)?;
            let config = LlamaConfig::stories15m();
            let mut model = Llama2LM::new(config);
            model.load_safetensors(&model_path)?;
            *lock = Some((model, tokenizer));
        }
        Ok(lock)
    }

    fn get_minilm(&self) -> Result<std::sync::MutexGuard<'_, Option<(BertModel, HfTokenizer)>>> {
        let mut lock = self.minilm.lock().unwrap();
        if lock.is_none() {
            let dir = self.checkpoints_dir.join("minilm");
            let model_path = dir.join("model.safetensors");
            let tok_path = dir.join("tokenizer.json");
            if !model_path.exists() || !tok_path.exists() {
                return Err(EngineError::InvalidArgument(format!(
                    "MiniLM checkpoint not found in {:?}.",
                    dir
                )));
            }
            eprintln!("[*] Loading MiniLM model from {:?}...", dir);
            let tokenizer = HfTokenizer::from_file(&tok_path)?;
            let config = BertConfig::all_minilm_l6_v2();
            let mut model = BertModel::new(config);
            model.load_safetensors(&model_path)?;
            *lock = Some((model, tokenizer));
        }
        Ok(lock)
    }

    fn get_modernbert(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<(ModernBertModel, HfTokenizer)>>> {
        let mut lock = self.modernbert.lock().unwrap();
        if lock.is_none() {
            let dir = self.checkpoints_dir.join("modernbert");
            let model_path = dir.join("model.safetensors");
            let tok_path = dir.join("tokenizer.json");
            if !model_path.exists() || !tok_path.exists() {
                return Err(EngineError::InvalidArgument(format!(
                    "ModernBERT checkpoint not found in {:?}.",
                    dir
                )));
            }
            eprintln!("[*] Loading ModernBERT model from {:?}...", dir);
            let tokenizer = HfTokenizer::from_file(&tok_path)?;
            let config = ModernBertConfig::base(tokenizer.vocab_size());
            let mut model = ModernBertModel::new(config);
            model.load_safetensors(&model_path)?;
            *lock = Some((model, tokenizer));
        }
        Ok(lock)
    }

    fn get_whisper(&self) -> Result<std::sync::MutexGuard<'_, Option<(Whisper, HfTokenizer)>>> {
        let mut lock = self.whisper.lock().unwrap();
        if lock.is_none() {
            let dir = self.checkpoints_dir.join("whisper");
            let model_path = dir.join("model.safetensors");
            let tok_path = dir.join("tokenizer.json");
            if !model_path.exists() || !tok_path.exists() {
                return Err(EngineError::InvalidArgument(format!(
                    "Whisper checkpoint not found in {:?}.",
                    dir
                )));
            }
            eprintln!("[*] Loading Whisper model from {:?}...", dir);
            let tokenizer = HfTokenizer::from_file(&tok_path)?;
            let config = WhisperConfig::whisper_tiny();
            let mut model = Whisper::new(config);
            model.load_safetensors(&model_path)?;
            *lock = Some((model, tokenizer));
        }
        Ok(lock)
    }

    fn preload(&self, targets: &[String]) {
        for target in targets {
            match target.as_str() {
                "gpt2" => {
                    drop(self.get_gpt2());
                }
                "tinyllamas" | "llama" => {
                    drop(self.get_tinyllamas());
                }
                "minilm" | "bert" => {
                    drop(self.get_minilm());
                }
                "modernbert" => {
                    drop(self.get_modernbert());
                }
                "whisper" => {
                    drop(self.get_whisper());
                }
                "all" => {
                    drop(self.get_gpt2());
                    drop(self.get_tinyllamas());
                    drop(self.get_minilm());
                    drop(self.get_modernbert());
                    drop(self.get_whisper());
                }
                other => {
                    eprintln!(
                        "[!] Warning: Unknown preload target '{}'. Available: gpt2, tinyllamas, minilm, modernbert, whisper, all",
                        other
                    );
                }
            }
        }
    }

    fn list_status(&self) -> Vec<Value> {
        let mut list = Vec::new();
        let models = [
            ("gpt2", self.gpt2.lock().unwrap().is_some()),
            ("tinyllamas", self.tinyllamas.lock().unwrap().is_some()),
            ("minilm", self.minilm.lock().unwrap().is_some()),
            ("modernbert", self.modernbert.lock().unwrap().is_some()),
            ("whisper", self.whisper.lock().unwrap().is_some()),
        ];
        for (name, loaded) in models {
            list.push(serde_json::json!({
                "id": name,
                "object": "model",
                "owned_by": "neural-network-engine",
                "status": if loaded { "loaded" } else { "available" },
            }));
        }
        list
    }
}

// ============================================================================
// Endpoint Request & Response Payloads
// ============================================================================

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    top_k: Option<usize>,
    #[serde(default = "default_stream")]
    stream: bool,
    seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    50
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_stream() -> bool {
    false
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    top_k: Option<usize>,
    #[serde(default = "default_stream")]
    stream: bool,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    model: Option<String>,
    input: Value, // string or array of strings
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TranscriptionJsonRequest {
    file: Option<String>,
    audio: Option<String>,
    model: Option<String>,
    language: Option<String>,
    task: Option<String>,
    temperature: Option<f32>,
}

// ============================================================================
// API Handler Implementations
// ============================================================================

fn handle_completions(
    req: &HttpRequest,
    stream: &mut TcpStream,
    registry: &Arc<ModelRegistry>,
    cors: bool,
) -> Result<()> {
    let payload: CompletionRequest = req.json_body()?;
    let model_name = payload
        .model
        .clone()
        .unwrap_or_else(|| "gpt2".to_string())
        .to_lowercase();
    let prompt = payload.prompt.unwrap_or_default();

    let mut rng = match payload.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let start_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let cmpl_id = format!("cmpl-{:x}", start_timestamp);

    if payload.stream {
        send_sse_headers(stream, cors)?;

        let stream_ptr = stream as *mut TcpStream;

        let model_name_for_cb = model_name.clone();
        let callback = move |_id: usize, piece: &str| -> Result<bool> {
            let chunk_obj = serde_json::json!({
                "id": cmpl_id,
                "object": "text_completion",
                "created": start_timestamp,
                "model": model_name_for_cb,
                "choices": [{
                    "text": piece,
                    "index": 0,
                    "finish_reason": null
                }]
            });
            let json_str = serde_json::to_string(&chunk_obj).unwrap();
            unsafe {
                if send_sse_chunk(&mut *stream_ptr, &json_str).is_err() {
                    return Ok(false);
                }
            }
            Ok(true)
        };

        match model_name.as_str() {
            "tinyllamas" | "llama" => {
                let mut guard = registry.get_tinyllamas()?;
                let (model, tokenizer) = guard.as_mut().unwrap();
                let prompt_tokens = tokenizer.encode(&prompt);
                model.generate_stream(
                    tokenizer,
                    &prompt_tokens,
                    payload.max_tokens,
                    payload.temperature,
                    payload.top_k,
                    Some(payload.top_p),
                    tokenizer.eos_token_id(),
                    &mut rng,
                    callback,
                )?;
            }
            _ => {
                let mut guard = registry.get_gpt2()?;
                let (model, tokenizer) = guard.as_mut().unwrap();
                let prompt_tokens = tokenizer.encode(&prompt);
                model.generate_stream(
                    tokenizer,
                    &prompt_tokens,
                    payload.max_tokens,
                    payload.temperature,
                    payload.top_k,
                    Some(payload.top_p),
                    tokenizer.eos_token_id(),
                    &mut rng,
                    callback,
                )?;
            }
        }

        send_sse_done(stream)?;
    } else {
        let mut generated_text = String::new();

        match model_name.as_str() {
            "tinyllamas" | "llama" => {
                let mut guard = registry.get_tinyllamas()?;
                let (model, tokenizer) = guard.as_mut().unwrap();
                let prompt_tokens = tokenizer.encode(&prompt);
                model.generate_stream(
                    tokenizer,
                    &prompt_tokens,
                    payload.max_tokens,
                    payload.temperature,
                    payload.top_k,
                    Some(payload.top_p),
                    tokenizer.eos_token_id(),
                    &mut rng,
                    |_id, piece| {
                        generated_text.push_str(piece);
                        Ok(true)
                    },
                )?;
            }
            _ => {
                let mut guard = registry.get_gpt2()?;
                let (model, tokenizer) = guard.as_mut().unwrap();
                let prompt_tokens = tokenizer.encode(&prompt);
                model.generate_stream(
                    tokenizer,
                    &prompt_tokens,
                    payload.max_tokens,
                    payload.temperature,
                    payload.top_k,
                    Some(payload.top_p),
                    tokenizer.eos_token_id(),
                    &mut rng,
                    |_id, piece| {
                        generated_text.push_str(piece);
                        Ok(true)
                    },
                )?;
            }
        }

        let resp_obj = serde_json::json!({
            "id": cmpl_id,
            "object": "text_completion",
            "created": start_timestamp,
            "model": model_name,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": "length"
            }]
        });
        send_json(stream, 200, "OK", &resp_obj, cors)?;
    }

    Ok(())
}

fn handle_chat_completions(
    req: &HttpRequest,
    stream: &mut TcpStream,
    registry: &Arc<ModelRegistry>,
    cors: bool,
) -> Result<()> {
    let payload: ChatCompletionRequest = req.json_body()?;
    let model_name = payload
        .model
        .clone()
        .unwrap_or_else(|| "gpt2".to_string())
        .to_lowercase();

    let mut formatted_prompt = String::new();
    for msg in &payload.messages {
        formatted_prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    formatted_prompt.push_str("assistant: ");

    let comp_req = HttpRequest {
        method: "POST".to_string(),
        path: "/v1/completions".to_string(),
        headers: req.headers.clone(),
        body: serde_json::to_vec(&serde_json::json!({
            "model": model_name,
            "prompt": formatted_prompt,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "top_k": payload.top_k,
            "stream": payload.stream,
            "seed": payload.seed,
        }))
        .unwrap(),
    };

    handle_completions(&comp_req, stream, registry, cors)
}

fn handle_embeddings(
    req: &HttpRequest,
    stream: &mut TcpStream,
    registry: &Arc<ModelRegistry>,
    cors: bool,
) -> Result<()> {
    let payload: EmbeddingRequest = req.json_body()?;
    let model_name = payload
        .model
        .unwrap_or_else(|| "minilm".to_string())
        .to_lowercase();

    let texts: Vec<String> = match payload.input {
        Value::String(s) => vec![s],
        Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => {
            return Err(EngineError::InvalidArgument(
                "Input must be a string or array of strings".to_string(),
            ))
        }
    };

    let mut embeddings = Vec::new();
    let mut total_tokens = 0;

    match model_name.as_str() {
        "modernbert" => {
            let mut guard = registry.get_modernbert()?;
            let (model, tokenizer) = guard.as_mut().unwrap();

            for (idx, text) in texts.iter().enumerate() {
                let tokens = tokenizer.encode(text);
                total_tokens += tokens.len();
                let cls_id = tokenizer
                    .cls_token_id()
                    .unwrap_or(tokenizer.bos_token_id().unwrap_or(0));
                let sep_id = tokenizer
                    .sep_token_id()
                    .unwrap_or(tokenizer.eos_token_id().unwrap_or(2));

                let mut input_ids = Vec::with_capacity(tokens.len() + 2);
                input_ids.push(cls_id);
                input_ids.extend(&tokens);
                input_ids.push(sep_id);

                let seq_len = input_ids.len();
                let input_tensor = Tensor::from_slice(
                    &input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[1, seq_len],
                    false,
                );

                let (_seq_out, pooled_out) = model.forward_model(&input_tensor)?;
                let contig = pooled_out.data().to_contiguous();
                let raw = contig.as_slice().to_vec();
                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let vec: Vec<f32> = raw.into_iter().map(|x| x / norm).collect();

                embeddings.push(serde_json::json!({
                    "object": "embedding",
                    "embedding": vec,
                    "index": idx
                }));
            }
        }
        _ => {
            let mut guard = registry.get_minilm()?;
            let (model, tokenizer) = guard.as_mut().unwrap();

            for (idx, text) in texts.iter().enumerate() {
                let tokens = tokenizer.encode(text);
                total_tokens += tokens.len();
                let cls_id = tokenizer.cls_token_id().unwrap_or(101);
                let sep_id = tokenizer.sep_token_id().unwrap_or(102);

                let mut input_ids = Vec::with_capacity(tokens.len() + 2);
                input_ids.push(cls_id);
                input_ids.extend(&tokens);
                input_ids.push(sep_id);

                let seq_len = input_ids.len();
                let input_tensor = Tensor::from_slice(
                    &input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[1, seq_len],
                    false,
                );
                let type_tensor = Tensor::zeros(&[1, seq_len], false);

                let (seq_out, _pooled_out) =
                    model.forward_bert(&input_tensor, Some(&type_tensor))?;
                let mean = seq_out.mean(1, false)?;
                let contig = mean.data().to_contiguous();
                let raw = contig.as_slice().to_vec();
                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let vec: Vec<f32> = raw.into_iter().map(|x| x / norm).collect();

                embeddings.push(serde_json::json!({
                    "object": "embedding",
                    "embedding": vec,
                    "index": idx
                }));
            }
        }
    }

    let resp_obj = serde_json::json!({
        "object": "list",
        "data": embeddings,
        "model": model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    });

    send_json(stream, 200, "OK", &resp_obj, cors)?;
    Ok(())
}

fn handle_audio_transcriptions(
    req: &HttpRequest,
    stream: &mut TcpStream,
    registry: &Arc<ModelRegistry>,
    cors: bool,
) -> Result<()> {
    let content_type = req.headers.get("content-type").cloned().unwrap_or_default();

    let mut wav_bytes: Option<Vec<u8>> = None;
    let mut language = "en".to_string();
    let mut task = "transcribe".to_string();

    if content_type.starts_with("multipart/form-data") {
        if let Some(form) = parse_multipart(&content_type, &req.body) {
            wav_bytes = form.file_bytes;
            if let Some(lang) = form.fields.get("language") {
                language = lang.to_lowercase();
            }
            if let Some(t) = form.fields.get("task") {
                task = t.to_lowercase();
            }
        }
    } else if content_type.starts_with("application/json") {
        let json_req: TranscriptionJsonRequest = req.json_body()?;
        if let Some(b64) = json_req.file.or(json_req.audio) {
            wav_bytes = base64_decode(&b64);
        }
        if let Some(lang) = json_req.language {
            language = lang.to_lowercase();
        }
        if let Some(t) = json_req.task {
            task = t.to_lowercase();
        }
    } else if content_type.starts_with("audio/wav")
        || content_type.starts_with("audio/x-wav")
        || req.body.starts_with(b"RIFF")
    {
        wav_bytes = Some(req.body.clone());
    }

    let audio_data = wav_bytes.ok_or_else(|| {
        EngineError::InvalidArgument(
            "Missing audio WAV payload (expected multipart/form-data 'file', base64 in JSON, or binary audio/wav)".to_string(),
        )
    })?;

    let mut wav = parse_wav_bytes(&audio_data)?;
    if wav.sample_rate != 16000 {
        wav.resample(16000);
    }

    let mel_raw = compute_whisper_mel_spectrogram(&wav.samples);
    let mel = Tensor::new(mel_raw.unsqueeze(0)?, false);

    let mut guard = registry.get_whisper()?;
    let (model, tokenizer) = guard.as_mut().unwrap();

    let memory = model.encode(&mel)?;

    // Prompt tokens
    let sot = tokenizer
        .token_to_id("<|startoftranscript|>")
        .unwrap_or(50258);
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50257);
    let lang_token = format!("<|{}|>", language);
    let lang_id = tokenizer
        .token_to_id(&lang_token)
        .or_else(|| tokenizer.token_to_id("<|en|>"))
        .unwrap_or(50259);
    let task_token = format!("<|{}|>", task);
    let task_id = tokenizer
        .token_to_id(&task_token)
        .or_else(|| tokenizer.token_to_id("<|transcribe|>"))
        .unwrap_or(50359);
    let notimestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap_or(50363);

    let prompt_tokens = vec![sot, lang_id, task_id, notimestamps];
    let prompt_len = prompt_tokens.len();
    let mut generated = prompt_tokens;

    for _ in 0..448 {
        let cur_len = generated.len();
        let tokens_raw = RawTensor::from_slice(
            &generated.iter().map(|&t| t as f32).collect::<Vec<_>>(),
            &[1, cur_len],
        );
        let tokens_tensor = Tensor::new(tokens_raw, false);

        let logits = model.decode(&tokens_tensor, &memory)?;
        let slice = logits.data().to_contiguous();
        let num_classes = model.config.vocab_size;
        let last_logits = &slice.as_slice()[(cur_len - 1) * num_classes..cur_len * num_classes];

        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in last_logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }

        if best_idx == eot {
            break;
        }
        generated.push(best_idx);
    }

    let transcription = tokenizer.decode(&generated[prompt_len..]);
    let cleaned = transcription.trim().to_string();

    let resp_obj = serde_json::json!({
        "text": cleaned
    });

    send_json(stream, 200, "OK", &resp_obj, cors)?;
    Ok(())
}

// ============================================================================
// Server Connection Dispatcher
// ============================================================================

fn handle_connection(
    mut stream: TcpStream,
    config: &ServerConfig,
    registry: &Arc<ModelRegistry>,
    start_time: Instant,
) {
    let req = match HttpRequest::parse(&mut stream) {
        Ok(r) => r,
        Err(_) => return,
    };

    // CORS preflight
    if req.method == "OPTIONS" {
        let _ = send_response(
            &mut stream,
            204,
            "No Content",
            "text/plain",
            b"",
            config.cors,
        );
        return;
    }

    // Bearer token authentication
    if let Some(ref expected_key) = config.api_key {
        let auth_header = req
            .headers
            .get("authorization")
            .cloned()
            .unwrap_or_default();
        let expected_header = format!("Bearer {}", expected_key);
        if auth_header != expected_header {
            let _ = send_error(
                &mut stream,
                401,
                "Unauthorized",
                "Invalid or missing API key in Authorization header",
                config.cors,
            );
            return;
        }
    }

    let res = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            let uptime = start_time.elapsed().as_secs();
            let status_models = registry.list_status();
            let resp = serde_json::json!({
                "status": "ok",
                "version": env!("CARGO_PKG_VERSION"),
                "uptime_seconds": uptime,
                "models": status_models
            });
            send_json(&mut stream, 200, "OK", &resp, config.cors).map_err(|e| e.into())
        }
        ("GET", "/v1/models") => {
            let models = registry.list_status();
            let resp = serde_json::json!({
                "object": "list",
                "data": models
            });
            send_json(&mut stream, 200, "OK", &resp, config.cors).map_err(|e| e.into())
        }
        ("POST", "/v1/completions") => handle_completions(&req, &mut stream, registry, config.cors),
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(&req, &mut stream, registry, config.cors)
        }
        ("POST", "/v1/embeddings") => handle_embeddings(&req, &mut stream, registry, config.cors),
        ("POST", "/v1/audio/transcriptions") => {
            handle_audio_transcriptions(&req, &mut stream, registry, config.cors)
        }
        _ => send_error(
            &mut stream,
            404,
            "Not Found",
            &format!("Endpoint '{}' not found", req.path),
            config.cors,
        )
        .map_err(|e| e.into()),
    };

    if let Err(e) = res {
        eprintln!("[!] Request error: {}", e);
        let _ = send_error(
            &mut stream,
            500,
            "Internal Server Error",
            &e.to_string(),
            config.cors,
        );
    }
}

// ============================================================================
// Main Server Entrypoint
// ============================================================================

fn main() -> Result<()> {
    let config = match parse_args() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Error: {}\n", err);
            print_help();
            process::exit(1);
        }
    };

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&bind_addr).map_err(|e| {
        EngineError::InvalidArgument(format!("Failed to bind to {}: {}", bind_addr, e))
    })?;

    let pool = ThreadPool::new(config.threads);
    let registry = Arc::new(ModelRegistry::new(config.checkpoints_dir.clone()));

    // Eager preload if requested
    if !config.preload.is_empty() {
        eprintln!("[*] Preloading requested models: {:?}...", config.preload);
        registry.preload(&config.preload);
        eprintln!("[*] Preload completed successfully.");
    }

    let start_time = Instant::now();
    let running = Arc::new(AtomicBool::new(true));

    eprintln!("====================================================================");
    eprintln!(" Neural Network Engine - OpenAI-Compatible HTTP Server");
    eprintln!(" Listening on: http://{}", bind_addr);
    eprintln!(" Worker Threads: {}", config.threads);
    eprintln!(" Checkpoints Dir: {:?}", config.checkpoints_dir);
    if config.api_key.is_some() {
        eprintln!(" Authentication: Bearer Token Enabled");
    }
    eprintln!(" CORS Enabled: {}", config.cors);
    eprintln!("====================================================================");
    eprintln!(" Endpoints ready:");
    eprintln!("   - GET  /health");
    eprintln!("   - GET  /v1/models");
    eprintln!("   - POST /v1/completions (supports stream=true SSE)");
    eprintln!("   - POST /v1/chat/completions");
    eprintln!("   - POST /v1/embeddings");
    eprintln!("   - POST /v1/audio/transcriptions");
    eprintln!("====================================================================\n");

    for stream_res in listener.incoming() {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        match stream_res {
            Ok(stream) => {
                let cfg = config.clone();
                let reg = Arc::clone(&registry);
                pool.execute(move || {
                    handle_connection(stream, &cfg, &reg, start_time);
                });
            }
            Err(e) => {
                eprintln!("[!] Connection error: {}", e);
            }
        }
    }

    Ok(())
}
