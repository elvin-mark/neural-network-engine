//! Standalone Extractive Question Answering & RAG CLI for Neural Network Engine.
//!
//! Powered by Dynamic TinyBERT fine-tuned on SQuAD (Reader) and all-MiniLM-L6-v2 (Retriever).
//! Supports direct span extraction, sliding window evaluation, and dense passage retrieval (RAG)
//! over chunked files and documents.

use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::bert::{BertConfig, BertForQuestionAnswering, BertModel};
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::Tensor;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CliArgs {
    checkpoint_dir: Option<PathBuf>,
    embedding_dir: Option<PathBuf>,
    context: Option<String>,
    context_file: Option<PathBuf>,
    question: Option<String>,
    positional: Vec<String>,
    max_answer_len: usize,
    rag: bool,
    chunk_size: usize,
    chunk_overlap: usize,
    top_chunks: usize,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            checkpoint_dir: None,
            embedding_dir: None,
            context: None,
            context_file: None,
            question: None,
            positional: Vec::new(),
            max_answer_len: 30,
            rag: false,
            chunk_size: 100,
            chunk_overlap: 20,
            top_chunks: 3,
        }
    }
}

fn print_help() {
    println!(
        r#"Neural Network Engine - Extractive Question Answering & RAG CLI

USAGE:
    cargo run --bin qa -- [OPTIONS] [CONTEXT] [QUESTION]

OPTIONS:
    -c, --context <TEXT>          Context passage containing the information
    -f, --context-file <FILE>     Read context passage from a text file
    -q, --question <TEXT>         Question to answer
        --checkpoint-dir <DIR>    QA model checkpoint directory [default: checkpoints/tinybert]
    -m, --max-answer-len <N>      Maximum span length for extracted answer [default: 30]

RAG & DENSE RETRIEVAL OPTIONS:
    -r, --rag, --retrieval        Enable RAG: chunk context, embed chunks, and retrieve best matches
        --chunk-size <N>          Maximum words per semantic chunk [default: 100]
        --chunk-overlap <N>       Overlap words between adjacent chunks [default: 20]
    -k, --top-chunks <N>          Number of highest-similarity chunks to evaluate [default: 3]
        --embedding-dir <DIR>     Dense embedding model checkpoint [default: checkpoints/minilm]

    -h, --help                    Print help information

EXAMPLES:
    # 1. One-shot Question Answering:
    cargo run --bin qa -- \
        --context "Paris is the capital and most populous city of France." \
        --question "What is the capital of France?"

    # 2. RAG on a text file (chunking, dense retrieval & QA):
    cargo run --bin qa -- \
        --context-file README.md \
        --question "What license is the engine released under?" \
        --rag

    # 3. Positional arguments:
    cargo run --bin qa -- \
        "Marie Curie was the first woman to win a Nobel Prize." \
        "Who was the first woman to win a Nobel Prize?"

    # 4. Interactive REPL session with RAG enabled:
    cargo run --bin qa -- --rag
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
        } else if arg == "-c" || arg == "--context" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --context".to_string());
            }
            args.context = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--context=") {
            args.context = Some(val.to_string());
        } else if arg == "-f" || arg == "--context-file" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --context-file".to_string());
            }
            args.context_file = Some(PathBuf::from(&raw_args[i]));
        } else if let Some(val) = arg.strip_prefix("--context-file=") {
            args.context_file = Some(PathBuf::from(val));
        } else if arg == "-q" || arg == "--question" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --question".to_string());
            }
            args.question = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--question=") {
            args.question = Some(val.to_string());
        } else if arg == "--checkpoint-dir" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --checkpoint-dir".to_string());
            }
            args.checkpoint_dir = Some(PathBuf::from(&raw_args[i]));
        } else if let Some(val) = arg.strip_prefix("--checkpoint-dir=") {
            args.checkpoint_dir = Some(PathBuf::from(val));
        } else if arg == "--embedding-dir" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --embedding-dir".to_string());
            }
            args.embedding_dir = Some(PathBuf::from(&raw_args[i]));
        } else if let Some(val) = arg.strip_prefix("--embedding-dir=") {
            args.embedding_dir = Some(PathBuf::from(val));
        } else if arg == "-r" || arg == "--rag" || arg == "--retrieval" {
            args.rag = true;
        } else if arg == "--chunk-size" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --chunk-size".to_string());
            }
            args.chunk_size = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid chunk-size: {}", e))?;
            args.rag = true;
        } else if let Some(val) = arg.strip_prefix("--chunk-size=") {
            args.chunk_size = val
                .parse()
                .map_err(|e| format!("Invalid chunk-size: {}", e))?;
            args.rag = true;
        } else if arg == "--chunk-overlap" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --chunk-overlap".to_string());
            }
            args.chunk_overlap = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid chunk-overlap: {}", e))?;
            args.rag = true;
        } else if let Some(val) = arg.strip_prefix("--chunk-overlap=") {
            args.chunk_overlap = val
                .parse()
                .map_err(|e| format!("Invalid chunk-overlap: {}", e))?;
            args.rag = true;
        } else if arg == "-k" || arg == "--top-chunks" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --top-chunks".to_string());
            }
            args.top_chunks = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid top-chunks: {}", e))?;
            args.rag = true;
        } else if let Some(val) = arg.strip_prefix("--top-chunks=") {
            args.top_chunks = val
                .parse()
                .map_err(|e| format!("Invalid top-chunks: {}", e))?;
            args.rag = true;
        } else if arg == "-m" || arg == "--max-answer-len" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --max-answer-len".to_string());
            }
            args.max_answer_len = raw_args[i]
                .parse()
                .map_err(|e| format!("Invalid max-answer-len: {}", e))?;
        } else if let Some(val) = arg.strip_prefix("--max-answer-len=") {
            args.max_answer_len = val
                .parse()
                .map_err(|e| format!("Invalid max-answer-len: {}", e))?;
        } else if arg.starts_with('-') {
            return Err(format!("Unknown option: {}", arg));
        } else {
            args.positional.push(arg.clone());
        }

        i += 1;
    }

    Ok(args)
}

fn softmax(slice: &[f32]) -> Vec<f32> {
    if slice.is_empty() {
        return Vec::new();
    }
    let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = slice.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum.max(1e-12)).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
}

fn chunk_document(text: &str, max_chunk_words: usize, overlap_words: usize) -> Vec<String> {
    let raw_paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk: Vec<&str> = Vec::new();
    let mut current_word_count = 0;

    for para in raw_paragraphs {
        let para_words: Vec<&str> = para.split_whitespace().collect();
        if para_words.len() > max_chunk_words {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.join("\n\n"));
                current_chunk.clear();
                current_word_count = 0;
            }
            let step = max_chunk_words.saturating_sub(overlap_words).max(1);
            let mut start = 0;
            while start < para_words.len() {
                let end = (start + max_chunk_words).min(para_words.len());
                chunks.push(para_words[start..end].join(" "));
                if end == para_words.len() {
                    break;
                }
                start += step;
            }
        } else if current_word_count + para_words.len() > max_chunk_words {
            chunks.push(current_chunk.join("\n\n"));
            current_chunk = vec![para];
            current_word_count = para_words.len();
        } else {
            current_chunk.push(para);
            current_word_count += para_words.len();
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join("\n\n"));
    }

    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(text.trim().to_string());
    }

    chunks
}

struct DenseRetriever {
    model: BertModel,
    tokenizer: HfTokenizer,
}

impl DenseRetriever {
    fn load(checkpoint_dir: &Path) -> Result<Self> {
        let model_path = checkpoint_dir.join("model.safetensors");
        let tokenizer_path = checkpoint_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(EngineError::InvalidArgument(format!(
                "Dense embedding tokenizer not found at {:?}",
                tokenizer_path
            )));
        }
        if !model_path.exists() {
            return Err(EngineError::InvalidArgument(format!(
                "Dense embedding model weights not found at {:?}",
                model_path
            )));
        }

        eprintln!(
            "Loading dense retriever tokenizer from {:?}...",
            tokenizer_path
        );
        let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

        eprintln!("Loading dense retriever weights from {:?}...", model_path);
        let config = BertConfig::all_minilm_l6_v2();
        let mut model = BertModel::new(config);
        model.load_safetensors(&model_path)?;

        Ok(Self { model, tokenizer })
    }

    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let mut tokens = self.tokenizer.encode(text);
        if tokens.len() > 510 {
            tokens.truncate(510);
        }
        let cls_id = self.tokenizer.cls_token_id().unwrap_or(101);
        let sep_id = self.tokenizer.sep_token_id().unwrap_or(102);

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

        let (seq_out, _) = self.model.forward_bert(&input_tensor, Some(&type_tensor))?;
        let mean = seq_out.mean(1, false)?;
        let contig = mean.data().to_contiguous();
        let raw = contig.as_slice();

        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(raw.iter().map(|x| x / norm).collect())
    }
}

fn highlight_answer(context: &str, answer: &str) -> String {
    if answer.is_empty() {
        return context.to_string();
    }

    let lower_ctx = context.to_lowercase();
    let lower_ans = answer.to_lowercase();
    let pos_opt = lower_ctx.find(&lower_ans).or_else(|| {
        lower_ans
            .split_whitespace()
            .next()
            .and_then(|w| lower_ctx.find(w))
    });

    if let Some(pos) = pos_opt {
        let mut end = pos + lower_ans.len();
        if end > context.len() {
            end = context.len();
        }
        while !context.is_char_boundary(end) && end < context.len() {
            end += 1;
        }

        // Snippet context around match: up to 120 chars before, 120 chars after
        let raw_start = pos.saturating_sub(120);
        let mut snippet_start = raw_start;
        while snippet_start > 0 && !context.is_char_boundary(snippet_start) {
            snippet_start -= 1;
        }

        let raw_end = (end + 120).min(context.len());
        let mut snippet_end = raw_end;
        while snippet_end < context.len() && !context.is_char_boundary(snippet_end) {
            snippet_end += 1;
        }

        let prefix = if snippet_start > 0 { "... " } else { "" };
        let suffix = if snippet_end < context.len() {
            " ..."
        } else {
            ""
        };

        let before = &context[snippet_start..pos];
        let matched = &context[pos..end];
        let after = &context[end..snippet_end];

        format!(
            "{}{}\x1b[1;32m[{}]\x1b[0m{}{}",
            prefix, before, matched, after, suffix
        )
    } else if context.len() > 300 {
        format!("... (Answer \"{}\" found in passage) ...", answer)
    } else {
        context.to_string()
    }
}

struct QaPrediction {
    answer: String,
    confidence: f32,
    /// Best span logit score minus the `[CLS]` "no-answer" score. Unlike softmax
    /// confidence, this is not diluted by context length, so it is comparable
    /// across chunks of different sizes.
    span_score: f32,
    start_token: usize,
    end_token: usize,
    chunk_index: Option<usize>,
    retrieval_score: Option<f32>,
}

#[allow(clippy::needless_range_loop)]
fn extract_answer_single_context(
    model: &BertForQuestionAnswering,
    tokenizer: &HfTokenizer,
    context: &str,
    question: &str,
    max_answer_len: usize,
) -> Result<QaPrediction> {
    let cls_id = tokenizer.cls_token_id().unwrap_or(101);
    let sep_id = tokenizer.sep_token_id().unwrap_or(102);

    let q_tokens = tokenizer.encode(question);
    let c_tokens = tokenizer.encode(context);

    let max_seq_len = 512;
    if q_tokens.len() + 3 >= max_seq_len {
        return Err(EngineError::InvalidArgument(
            "Question is too long for model context window (exceeds 512 tokens)".to_string(),
        ));
    }
    let max_c_len = max_seq_len - q_tokens.len() - 3;
    let stride = (max_c_len / 2).max(64);

    let mut windows = Vec::new();
    let mut win_start = 0;
    while win_start < c_tokens.len() {
        let win_end = (win_start + max_c_len).min(c_tokens.len());
        windows.push((win_start, win_end));
        if win_end == c_tokens.len() {
            break;
        }
        win_start += stride;
    }
    if windows.is_empty() {
        windows.push((0, 0));
    }

    let mut global_best_score = f32::NEG_INFINITY;
    let mut global_best_span = (0, 0);
    let mut global_best_prob = 0.0f32;
    let mut global_best_span_score = f32::NEG_INFINITY;
    let mut global_best_tokens = Vec::new();

    for &(w_start, w_end) in &windows {
        let window_tokens = &c_tokens[w_start..w_end];
        let mut input_ids = Vec::with_capacity(q_tokens.len() + window_tokens.len() + 3);
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
        for &tok in window_tokens {
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

        let start_probs = softmax(start_s);
        let end_probs = softmax(end_s);

        let null_score = start_s[0] + end_s[0];

        for s in context_start_idx..context_end_idx {
            let limit = std::cmp::min(s + max_answer_len, context_end_idx);
            for e in s..limit {
                let score = start_s[s] + end_s[e];
                if score > global_best_score {
                    global_best_score = score;
                    global_best_span_score = score - null_score;
                    let local_s = s - context_start_idx;
                    let local_e = e - context_start_idx;
                    global_best_span = (w_start + local_s, w_start + local_e);
                    global_best_prob = start_probs[s] * end_probs[e];
                    global_best_tokens = input_ids[s..=e].to_vec();
                }
            }
        }
    }

    let answer = tokenizer.decode(&global_best_tokens);

    Ok(QaPrediction {
        answer: answer.trim().to_string(),
        confidence: global_best_prob,
        span_score: global_best_span_score,
        start_token: global_best_span.0,
        end_token: global_best_span.1,
        chunk_index: None,
        retrieval_score: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn extract_answer_rag(
    model: &BertForQuestionAnswering,
    tokenizer: &HfTokenizer,
    retriever: &DenseRetriever,
    context: &str,
    question: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    top_chunks_k: usize,
    max_answer_len: usize,
) -> Result<QaPrediction> {
    let chunks = chunk_document(context, chunk_size, chunk_overlap);
    if chunks.is_empty() {
        return extract_answer_single_context(model, tokenizer, context, question, max_answer_len);
    }

    eprintln!(
        "[*] [RAG] Segmented passage into {} chunks (chunk_size: {} words, overlap: {} words)",
        chunks.len(),
        chunk_size,
        chunk_overlap
    );

    // 1. Embed question and all chunks
    let q_emb = retriever.encode(question)?;

    let mut ranked_chunks: Vec<(f32, usize, &str)> = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let c_emb = retriever.encode(chunk)?;
        let sim = cosine_similarity(&q_emb, &c_emb);
        ranked_chunks.push((sim, idx, chunk.as_str()));
    }

    // Sort descending by similarity score
    ranked_chunks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("[*] [RAG] Top Retrieved Candidate Chunks:");
    for (i, &(score, idx, chunk)) in ranked_chunks.iter().take(top_chunks_k).enumerate() {
        let preview: String = chunk.chars().take(80).collect();
        eprintln!(
            "    #{}: Score {:+.4} | Chunk #{} -> \"{}...\"",
            i + 1,
            score,
            idx,
            preview.replace('\n', " ")
        );
    }

    // 2. Evaluate top K chunks with QA Reader
    let mut best_pred: Option<QaPrediction> = None;
    let mut best_score = f32::NEG_INFINITY;

    for &(sim_score, chunk_idx, chunk_text) in ranked_chunks.iter().take(top_chunks_k) {
        let mut pred =
            extract_answer_single_context(model, tokenizer, chunk_text, question, max_answer_len)?;
        pred.chunk_index = Some(chunk_idx);
        pred.retrieval_score = Some(sim_score);

        // Rank candidates primarily by the reader's span score (best span logits
        // minus the [CLS] null-answer logits). Softmax confidence is diluted by
        // chunk length and is not comparable across chunks, whereas logit margins
        // are. Retrieval similarity (in [-1, 1]) acts as a small tie-breaker.
        let candidate_score = pred.span_score + sim_score * 2.0;

        eprintln!(
            "    - Candidate Chunk #{}: Ans=\"{}\", Conf={:.1}%, SpanScore={:+.3}, RetrScore={:+.4} -> CombinedScore={:+.3}",
            chunk_idx,
            pred.answer,
            pred.confidence * 100.0,
            pred.span_score,
            sim_score,
            candidate_score
        );

        if candidate_score > best_score && !pred.answer.is_empty() {
            best_score = candidate_score;
            best_pred = Some(pred);
        }
    }

    best_pred.ok_or_else(|| {
        EngineError::InvalidArgument("Failed to extract answer from retrieved chunks".to_string())
    })
}

fn print_result(context: &str, question: &str, pred: &QaPrediction) {
    let highlighted = highlight_answer(context, &pred.answer);

    println!("\n==============================================================");
    println!("Question:          \"{}\"", question);
    println!("Extracted Answer:  \x1b[1;32m{}\x1b[0m", pred.answer);
    println!("Confidence:        {:.1}%", pred.confidence * 100.0);

    if let (Some(idx), Some(score)) = (pred.chunk_index, pred.retrieval_score) {
        println!("Retrieved Chunk:   #{} (Similarity: {:+.4})", idx, score);
    } else {
        println!(
            "Context Token Span: {}..{}",
            pred.start_token, pred.end_token
        );
    }

    println!("--------------------------------------------------------------");
    println!("Context Highlight:\n  {}", highlighted);
    println!("==============================================================\n");
}

fn run_interactive_repl(
    model: &BertForQuestionAnswering,
    tokenizer: &HfTokenizer,
    retriever: Option<&DenseRetriever>,
    initial_context: Option<String>,
    args: &CliArgs,
) -> Result<()> {
    println!("==============================================================");
    println!("Neural Network Engine - Interactive Question Answering Console");
    println!("Model:     Dynamic TinyBERT (SQuAD Extractive QA)");
    if retriever.is_some() {
        println!("Retriever: all-MiniLM-L6-v2 (Dense Passage Retrieval enabled)");
        println!(
            "Chunking:  size={} words, overlap={} words, top_k={}",
            args.chunk_size, args.chunk_overlap, args.top_chunks
        );
    }
    println!("==============================================================");
    println!("Commands:");
    println!("  :context <text>    Set a new context passage");
    println!("  :file <path>       Load context passage from a file");
    println!("  :clear             Clear current context");
    println!("  /quit or /exit     Exit the console");
    println!("==============================================================\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut current_context = initial_context.unwrap_or_default();

    loop {
        if current_context.is_empty() {
            println!("Please enter or paste your context passage below (or :file <path>):");
            print!("context >>> ");
            io::stdout().flush()?;

            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed == "/quit" || trimmed == "/exit" {
                break;
            }
            if let Some(path_str) = trimmed.strip_prefix(":file ") {
                let p = Path::new(path_str.trim());
                match fs::read_to_string(p) {
                    Ok(content) => {
                        current_context = content.trim().to_string();
                        println!(
                            "Loaded context from {:?} ({} chars)\n",
                            p,
                            current_context.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("Error reading file {:?}: {}\n", p, e);
                    }
                }
                continue;
            }
            current_context = trimmed.to_string();
            if current_context.is_empty() {
                continue;
            }
            println!(
                "Context set ({} chars). You can now ask questions!\n",
                current_context.len()
            );
        }

        print!("question >>> ");
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
        if trimmed == ":clear" {
            current_context.clear();
            println!("Context cleared.\n");
            continue;
        }
        if let Some(ctx) = trimmed.strip_prefix(":context ") {
            current_context = ctx.trim().to_string();
            println!("New context set ({} chars).\n", current_context.len());
            continue;
        }
        if let Some(path_str) = trimmed.strip_prefix(":file ") {
            let p = Path::new(path_str.trim());
            match fs::read_to_string(p) {
                Ok(content) => {
                    current_context = content.trim().to_string();
                    println!(
                        "Loaded context from {:?} ({} chars)\n",
                        p,
                        current_context.len()
                    );
                }
                Err(e) => {
                    eprintln!("Error reading file {:?}: {}\n", p, e);
                }
            }
            continue;
        }

        let pred = if let Some(r) = retriever {
            extract_answer_rag(
                model,
                tokenizer,
                r,
                &current_context,
                trimmed,
                args.chunk_size,
                args.chunk_overlap,
                args.top_chunks,
                args.max_answer_len,
            )?
        } else {
            extract_answer_single_context(
                model,
                tokenizer,
                &current_context,
                trimmed,
                args.max_answer_len,
            )?
        };

        print_result(&current_context, trimmed, &pred);
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Use --help for usage information.");
            std::process::exit(1);
        }
    };

    let checkpoint_dir = args
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("checkpoints/tinybert"));

    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Tokenizer file not found at {:?}. Please ensure checkpoint directory exists.",
            tokenizer_path
        )));
    }
    if !model_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Model SafeTensors weights not found at {:?}. Please ensure checkpoint exists.",
            model_path
        )));
    }

    eprintln!("Loading TinyBERT QA tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading Dynamic TinyBERT QA model from {:?}...", model_path);
    let config = BertConfig::dynamic_tinybert();
    let mut model = BertForQuestionAnswering::new(config);
    model.load_safetensors(&model_path)?;
    eprintln!("Model loaded successfully!\n");

    // Initialize Dense Retriever if RAG is requested
    let retriever = if args.rag {
        let emb_dir = args
            .embedding_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("checkpoints/minilm"));
        Some(DenseRetriever::load(&emb_dir)?)
    } else {
        None
    };

    // Resolve context: from --context, --context-file, or positional[0]
    let context = if let Some(ref c) = args.context {
        Some(c.clone())
    } else if let Some(ref f) = args.context_file {
        Some(fs::read_to_string(f).map_err(|e| {
            EngineError::InvalidArgument(format!("Failed to read context file {:?}: {}", f, e))
        })?)
    } else if !args.positional.is_empty() {
        Some(args.positional[0].clone())
    } else {
        None
    };

    // Resolve question: from --question or positional[1]
    let question = if let Some(ref q) = args.question {
        Some(q.clone())
    } else if args.positional.len() >= 2 {
        Some(args.positional[1].clone())
    } else {
        None
    };

    match (context, question) {
        (Some(c), Some(q)) => {
            let pred = if let Some(ref r) = retriever {
                extract_answer_rag(
                    &model,
                    &tokenizer,
                    r,
                    &c,
                    &q,
                    args.chunk_size,
                    args.chunk_overlap,
                    args.top_chunks,
                    args.max_answer_len,
                )?
            } else {
                extract_answer_single_context(&model, &tokenizer, &c, &q, args.max_answer_len)?
            };
            print_result(&c, &q, &pred);
        }
        (Some(c), None) => {
            run_interactive_repl(&model, &tokenizer, retriever.as_ref(), Some(c), &args)?;
        }
        (None, Some(q)) => {
            println!("Question specified: \"{}\"", q);
            println!("Please provide the context passage to search for the answer.");
            run_interactive_repl(&model, &tokenizer, retriever.as_ref(), None, &args)?;
        }
        (None, None) => {
            run_interactive_repl(&model, &tokenizer, retriever.as_ref(), None, &args)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_document_basic() {
        let text = "Paragraph one with some words.\n\nParagraph two with some additional words.\n\nParagraph three with more content.";
        let chunks = chunk_document(text, 10, 2);
        assert!(!chunks.is_empty());
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunk_document_long_paragraph() {
        let long_para = (0..100)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_document(&long_para, 30, 5);
        assert!(chunks.len() >= 4);
        assert!(chunks[0].starts_with("word0 word1"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }
}
