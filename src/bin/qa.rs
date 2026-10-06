//! Standalone Extractive Question Answering CLI for Neural Network Engine.
//!
//! Powered by Dynamic TinyBERT fine-tuned on SQuAD, extracting exact answer spans
//! from context paragraphs with confidence scores and context highlighting.

use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::bert::{BertConfig, BertForQuestionAnswering};
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::Tensor;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CliArgs {
    checkpoint_dir: Option<PathBuf>,
    context: Option<String>,
    context_file: Option<PathBuf>,
    question: Option<String>,
    positional: Vec<String>,
    max_answer_len: usize,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            checkpoint_dir: None,
            context: None,
            context_file: None,
            question: None,
            positional: Vec::new(),
            max_answer_len: 30,
        }
    }
}

fn print_help() {
    println!(
        r#"Neural Network Engine - Extractive Question Answering CLI

USAGE:
    cargo run --bin qa -- [OPTIONS] [CONTEXT] [QUESTION]

OPTIONS:
    -c, --context <TEXT>          Context passage containing the information
    -f, --context-file <FILE>     Read context passage from a text file
    -q, --question <TEXT>         Question to answer
        --checkpoint-dir <DIR>    Custom directory containing model.safetensors and tokenizer.json
                                  [default: checkpoints/tinybert]
    -m, --max-answer-len <N>      Maximum span length for extracted answer [default: 30]
    -h, --help                    Print help information

EXAMPLES:
    # 1. One-shot Question Answering:
    cargo run --bin qa -- \
        --context "Paris is the capital and most populous city of France." \
        --question "What is the capital of France?"

    # 2. Positional arguments:
    cargo run --bin qa -- \
        "Marie Curie was the first woman to win a Nobel Prize." \
        "Who was the first woman to win a Nobel Prize?"

    # 3. Read context from a text file:
    cargo run --bin qa --context-file README.md --question "What features does the engine support?"

    # 4. Interactive REPL session:
    cargo run --bin qa
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
    start_token: usize,
    end_token: usize,
}

#[allow(clippy::needless_range_loop)]
fn extract_answer(
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

        for s in context_start_idx..context_end_idx {
            let limit = std::cmp::min(s + max_answer_len, context_end_idx);
            for e in s..limit {
                let score = start_s[s] + end_s[e];
                if score > global_best_score {
                    global_best_score = score;
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
        start_token: global_best_span.0,
        end_token: global_best_span.1,
    })
}

fn print_result(context: &str, question: &str, pred: &QaPrediction) {
    let highlighted = highlight_answer(context, &pred.answer);

    println!("\n==============================================================");
    println!("Question:          \"{}\"", question);
    println!("Extracted Answer:  \x1b[1;32m{}\x1b[0m", pred.answer);
    println!("Confidence:        {:.1}%", pred.confidence * 100.0);
    println!(
        "Context Token Span: {}..{}",
        pred.start_token, pred.end_token
    );
    println!("--------------------------------------------------------------");
    println!("Context Highlight:\n  {}", highlighted);
    println!("==============================================================\n");
}

fn run_interactive_repl(
    model: &BertForQuestionAnswering,
    tokenizer: &HfTokenizer,
    initial_context: Option<String>,
    max_answer_len: usize,
) -> Result<()> {
    println!("==============================================================");
    println!("Neural Network Engine - Interactive Question Answering Console");
    println!("Model: Dynamic TinyBERT (SQuAD Extractive QA)");
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

        let pred = extract_answer(model, tokenizer, &current_context, trimmed, max_answer_len)?;
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

    // Resolve context: from --context, --context-file, or positional[0]
    let context = if let Some(c) = args.context {
        Some(c)
    } else if let Some(f) = args.context_file {
        Some(fs::read_to_string(&f).map_err(|e| {
            EngineError::InvalidArgument(format!("Failed to read context file {:?}: {}", f, e))
        })?)
    } else if !args.positional.is_empty() {
        Some(args.positional[0].clone())
    } else {
        None
    };

    // Resolve question: from --question or positional[1]
    let question = if let Some(q) = args.question {
        Some(q)
    } else if args.positional.len() >= 2 {
        Some(args.positional[1].clone())
    } else {
        None
    };

    match (context, question) {
        (Some(c), Some(q)) => {
            let pred = extract_answer(&model, &tokenizer, &c, &q, args.max_answer_len)?;
            print_result(&c, &q, &pred);
        }
        (Some(c), None) => {
            run_interactive_repl(&model, &tokenizer, Some(c), args.max_answer_len)?;
        }
        (None, Some(q)) => {
            println!("Question specified: \"{}\"", q);
            println!("Please provide the context passage to search for the answer.");
            run_interactive_repl(&model, &tokenizer, None, args.max_answer_len)?;
        }
        (None, None) => {
            run_interactive_repl(&model, &tokenizer, None, args.max_answer_len)?;
        }
    }

    Ok(())
}
