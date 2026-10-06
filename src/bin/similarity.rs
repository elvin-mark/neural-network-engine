//! Semantic Sentence Similarity & Text Embedding CLI for Neural Network Engine.
//!
//! Computes high-dimensional dense embeddings for sentences using BERT (e.g. all-MiniLM-L6-v2)
//! or ModernBERT, and evaluates cosine similarity for semantic search and pairwise comparison.

use neural_network_engine::error::{EngineError, Result};
use neural_network_engine::models::bert::{BertConfig, BertModel};
use neural_network_engine::models::modern_bert::{ModernBertConfig, ModernBertModel};
use neural_network_engine::tokenizer::HfTokenizer;
use neural_network_engine::Tensor;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Pooling {
    #[default]
    Mean,
    Cls,
    Pooler,
}

impl Pooling {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mean" | "average" => Some(Self::Mean),
            "cls" => Some(Self::Cls),
            "pooler" | "dense" => Some(Self::Pooler),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Mean => "Mean Pooling (Sentence-Transformers standard)",
            Self::Cls => "[CLS] Token Embedding",
            Self::Pooler => "Dense Pooler Projection",
        }
    }
}

enum EmbeddingEngine {
    Bert(Box<BertModel>),
    ModernBert(ModernBertModel),
}

impl EmbeddingEngine {
    fn encode(&self, tokenizer: &HfTokenizer, text: &str, pooling: Pooling) -> Result<Vec<f32>> {
        match self {
            EmbeddingEngine::Bert(model) => {
                let tokens = tokenizer.encode(text);
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

                let (seq_out, pooled_out) =
                    model.forward_bert(&input_tensor, Some(&type_tensor))?;

                let raw = match pooling {
                    Pooling::Mean => {
                        let mean = seq_out.mean(1, false)?;
                        let contig = mean.data().to_contiguous();
                        contig.as_slice().to_vec()
                    }
                    Pooling::Cls => {
                        let cls = seq_out.slice(1, 0, 1)?;
                        let contig = cls.data().to_contiguous();
                        contig.as_slice().to_vec()
                    }
                    Pooling::Pooler => {
                        let contig = pooled_out.data().to_contiguous();
                        contig.as_slice().to_vec()
                    }
                };

                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                Ok(raw.into_iter().map(|x| x / norm).collect())
            }
            EmbeddingEngine::ModernBert(model) => {
                let tokens = tokenizer.encode(text);
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

                let (seq_out, pooled_out) = model.forward_model(&input_tensor)?;

                let raw = match pooling {
                    Pooling::Mean | Pooling::Pooler => {
                        let contig = pooled_out.data().to_contiguous();
                        contig.as_slice().to_vec()
                    }
                    Pooling::Cls => {
                        let cls = seq_out.slice(1, 0, 1)?;
                        let contig = cls.data().to_contiguous();
                        contig.as_slice().to_vec()
                    }
                };

                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                Ok(raw.into_iter().map(|x| x / norm).collect())
            }
        }
    }
}

#[derive(Debug)]
struct CliArgs {
    model: String,
    checkpoint_dir: Option<PathBuf>,
    pooling: Pooling,
    query: Option<String>,
    candidates: Vec<String>,
    positional: Vec<String>,
    top_k: Option<usize>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model: "minilm".to_string(),
            checkpoint_dir: None,
            pooling: Pooling::default(),
            query: None,
            candidates: Vec::new(),
            positional: Vec::new(),
            top_k: None,
        }
    }
}

fn print_help() {
    println!(
        r#"Neural Network Engine - Semantic Sentence Similarity CLI

USAGE:
    cargo run --bin similarity -- [OPTIONS] [SENTENCES...]

OPTIONS:
    -m, --model <MODEL>           Embedding model architecture:
                                  minilm, bert, modern_bert
                                  [default: minilm]
    -c, --checkpoint-dir <DIR>    Custom directory containing model.safetensors and tokenizer.json
                                  [default: checkpoints/<model>]
    -p, --pooling <STRATEGY>      Pooling method: mean, cls, pooler [default: mean]
    -q, --query <TEXT>            Query sentence for 1-to-many semantic search ranking
    -C, --candidate <TEXT>        Candidate sentence to rank against query (can be repeated)
    -k, --top-k <N>               Limit displayed candidate rankings to top N results
    -h, --help                    Print help information

EXAMPLES:
    # 1. Pairwise sentence comparison:
    cargo run --bin similarity -- "The dog plays in the yard." "A puppy is running outdoors."

    # 2. Semantic search ranking (query against candidates):
    cargo run --bin similarity --query "Deep learning in Rust" \
        -C "High performance autograd neural network engine" \
        -C "Baking homemade sourdough bread" \
        -C "Rust memory safety and concurrency"

    # 3. Interactive console (if no sentences are passed):
    cargo run --bin similarity
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
        } else if arg == "-p" || arg == "--pooling" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --pooling".to_string());
            }
            args.pooling = Pooling::from_str(&raw_args[i])
                .ok_or_else(|| format!("Invalid pooling strategy: {}", raw_args[i]))?;
        } else if let Some(val) = arg.strip_prefix("--pooling=") {
            args.pooling = Pooling::from_str(val)
                .ok_or_else(|| format!("Invalid pooling strategy: {}", val))?;
        } else if arg == "-q" || arg == "--query" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --query".to_string());
            }
            args.query = Some(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--query=") {
            args.query = Some(val.to_string());
        } else if arg == "-C" || arg == "--candidate" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --candidate".to_string());
            }
            args.candidates.push(raw_args[i].clone());
        } else if let Some(val) = arg.strip_prefix("--candidate=") {
            args.candidates.push(val.to_string());
        } else if arg == "-k" || arg == "--top-k" {
            i += 1;
            if i >= raw_args.len() {
                return Err("Missing value for --top-k".to_string());
            }
            args.top_k = Some(
                raw_args[i]
                    .parse()
                    .map_err(|e| format!("Invalid top-k: {}", e))?,
            );
        } else if let Some(val) = arg.strip_prefix("--top-k=") {
            args.top_k = Some(val.parse().map_err(|e| format!("Invalid top-k: {}", e))?);
        } else if arg.starts_with('-') {
            return Err(format!("Unknown option: {}", arg));
        } else {
            args.positional.push(arg.clone());
        }

        i += 1;
    }

    Ok(args)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
}

fn render_bar(score: f32, width: usize) -> String {
    let clamped = score.clamp(0.0, 1.0);
    let filled = (clamped * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn semantic_match_desc(score: f32) -> &'static str {
    if score >= 0.85 {
        "Extremely high similarity / Paraphrase"
    } else if score >= 0.70 {
        "High similarity / Strongly related"
    } else if score >= 0.50 {
        "Moderate similarity / Common theme"
    } else if score >= 0.30 {
        "Low similarity / Weak topical overlap"
    } else {
        "Unrelated / Orthogonal context"
    }
}

fn load_engine(
    model_name: &str,
    checkpoint_dir: &Path,
) -> Result<(EmbeddingEngine, HfTokenizer, String)> {
    let model_path = checkpoint_dir.join("model.safetensors");
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Tokenizer file not found at {:?}. Please ensure checkpoint directory exists.",
            tokenizer_path
        )));
    }

    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    if !model_path.exists() {
        return Err(EngineError::InvalidArgument(format!(
            "Model SafeTensors weights not found at {:?}. Please convert the model first.",
            model_path
        )));
    }

    match model_name {
        "modern_bert" => {
            eprintln!("Loading ModernBERT model weights from {:?}...", model_path);
            let config = ModernBertConfig::base(tokenizer.vocab_size());
            let mut model = ModernBertModel::new(config);
            model.load_safetensors(&model_path)?;
            let desc = format!("ModernBERT (d_model={})", model.config.d_model);
            Ok((EmbeddingEngine::ModernBert(model), tokenizer, desc))
        }
        _ => {
            // Default to BERT / all-MiniLM-L6-v2
            eprintln!("Loading BERT model weights from {:?}...", model_path);
            let config = BertConfig::all_minilm_l6_v2();
            let mut model = BertModel::new(config);
            model.load_safetensors(&model_path)?;
            let desc = format!("BERT / all-MiniLM-L6-v2 (d_model={})", model.config.d_model);
            Ok((EmbeddingEngine::Bert(Box::new(model)), tokenizer, desc))
        }
    }
}

fn compare_two_sentences(
    engine: &EmbeddingEngine,
    tokenizer: &HfTokenizer,
    sent_a: &str,
    sent_b: &str,
    pooling: Pooling,
) -> Result<()> {
    let emb_a = engine.encode(tokenizer, sent_a, pooling)?;
    let emb_b = engine.encode(tokenizer, sent_b, pooling)?;
    let sim = cosine_similarity(&emb_a, &emb_b);
    let pct = (sim * 100.0).max(0.0);

    println!("\nPairwise Sentence Comparison:");
    println!("  Sentence A: \"{}\"", sent_a);
    println!("  Sentence B: \"{}\"", sent_b);
    println!();
    println!("  Cosine Similarity: {:+.4} ({:.1}%)", sim, pct);
    println!("  Match Level:       {}", semantic_match_desc(sim));
    println!("  Visual Score:      {} {:.1}%\n", render_bar(sim, 24), pct);

    Ok(())
}

fn rank_candidates(
    engine: &EmbeddingEngine,
    tokenizer: &HfTokenizer,
    query: &str,
    candidates: &[String],
    pooling: Pooling,
    top_k: Option<usize>,
) -> Result<()> {
    if candidates.is_empty() {
        println!("No candidate sentences provided for query: \"{}\"", query);
        return Ok(());
    }

    let q_emb = engine.encode(tokenizer, query, pooling)?;

    let mut scored: Vec<(f32, &str)> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let c_emb = engine.encode(tokenizer, cand, pooling)?;
        let sim = cosine_similarity(&q_emb, &c_emb);
        scored.push((sim, cand.as_str()));
    }

    // Sort descending by similarity score
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(k) = top_k {
        scored.truncate(k);
    }

    println!("\nQuery: \"{}\"", query);
    println!("Top Ranked Semantic Matches:");
    println!(
        "{:<4}  {:<8}  {:<26}  Candidate Sentence",
        "Rank", "Score", "Similarity Bar"
    );
    println!("{:-<80}", "");

    for (idx, (score, cand)) in scored.iter().enumerate() {
        let pct = (score * 100.0).max(0.0);
        let bar = render_bar(*score, 20);
        println!(
            "#{:<3}  {:+0.4}  {} {:>5.1}%  \"{}\"",
            idx + 1,
            score,
            bar,
            pct,
            cand
        );
    }
    println!();

    Ok(())
}

fn run_interactive_repl(
    engine: &EmbeddingEngine,
    tokenizer: &HfTokenizer,
    model_desc: &str,
    pooling: Pooling,
) -> Result<()> {
    println!("==============================================================");
    println!("Neural Network Engine - Semantic Sentence Similarity Console");
    println!("Model:   {}", model_desc);
    println!("Pooling: {}", pooling.as_str());
    println!("==============================================================");
    println!("Commands:");
    println!("  1. Pairwise:   Enter two sentences separated by ' | '");
    println!("                 e.g. 'The dog barked | A puppy was making noise'");
    println!("  2. Query Mode: Set a persistent query using ':q <query text>'");
    println!("                 e.g. ':q Rust machine learning'");
    println!("                 Then enter candidate sentences one by one to score!");
    println!("  3. Reset:      Type ':clear' to reset active query");
    println!("  4. Exit:       Type '/quit' or '/exit' to exit");
    println!("==============================================================\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut active_query: Option<String> = None;
    let mut active_query_emb: Option<Vec<f32>> = None;

    loop {
        if let Some(ref q) = active_query {
            print!("[query: \"{}\"] >>> ", q);
        } else {
            print!(">>> ");
        }
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
            active_query = None;
            active_query_emb = None;
            println!("Query cleared. Switched back to pairwise mode.\n");
            continue;
        }

        if let Some(query_text) = trimmed.strip_prefix(":q ") {
            let q_str = query_text.trim();
            if q_str.is_empty() {
                println!("Empty query specified.\n");
                continue;
            }
            let emb = engine.encode(tokenizer, q_str, pooling)?;
            println!("Active query set: \"{}\"\n", q_str);
            active_query = Some(q_str.to_string());
            active_query_emb = Some(emb);
            continue;
        }

        if trimmed.contains('|') {
            let parts: Vec<&str> = trimmed.split('|').map(|s| s.trim()).collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                compare_two_sentences(engine, tokenizer, parts[0], parts[1], pooling)?;
                continue;
            }
        }

        if let (Some(_), Some(ref q_emb)) = (&active_query, &active_query_emb) {
            let c_emb = engine.encode(tokenizer, trimmed, pooling)?;
            let sim = cosine_similarity(q_emb, &c_emb);
            let pct = (sim * 100.0).max(0.0);
            println!(
                "  Score: {:+.4}  {} {:>5.1}%  | Match: {}\n",
                sim,
                render_bar(sim, 20),
                pct,
                semantic_match_desc(sim)
            );
        } else {
            println!("Tip: Separate two sentences with ' | ' to compare, or use ':q <text>' to set a search query.\n");
        }
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

    let default_dir = match args.model.as_str() {
        "modern_bert" => "checkpoints/modern_bert",
        _ => "checkpoints/minilm",
    };

    let checkpoint_dir = args
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_dir));

    let (engine, tokenizer, desc) = load_engine(&args.model, &checkpoint_dir)?;

    // Case 1: Query with explicit or positional candidates
    if let Some(query) = args.query {
        let mut candidates = args.candidates;
        candidates.extend(args.positional);
        rank_candidates(
            &engine,
            &tokenizer,
            &query,
            &candidates,
            args.pooling,
            args.top_k,
        )?;
        return Ok(());
    }

    // Case 2: Pairwise comparison with exactly two positional arguments
    if args.positional.len() == 2 {
        compare_two_sentences(
            &engine,
            &tokenizer,
            &args.positional[0],
            &args.positional[1],
            args.pooling,
        )?;
        return Ok(());
    }

    // Case 3: More than 2 positional arguments: treat first as query and remainder as candidates
    if args.positional.len() > 2 {
        let query = &args.positional[0];
        let candidates = &args.positional[1..];
        rank_candidates(
            &engine,
            &tokenizer,
            query,
            candidates,
            args.pooling,
            args.top_k,
        )?;
        return Ok(());
    }

    // Case 4: No positional arguments and no query: start Interactive REPL
    run_interactive_repl(&engine, &tokenizer, &desc, args.pooling)?;

    Ok(())
}
