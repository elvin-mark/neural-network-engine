//! Native zero-dependency Hugging Face `tokenizer.json` parser, encoder, and decoder.
//!
//! Supports:
//! - Byte-Level BPE (GPT-2, Whisper)
//! - SentencePiece / LLaMA-style BPE (TinyStories, LLaMA)
//! - WordPiece (BERT, TinyBERT)

use crate::error::{EngineError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Tokenizer flavor parsed from `tokenizer.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerKind {
    /// Byte-Level BPE with GPT-2 byte mapping (e.g. GPT-2, Whisper).
    ByteLevelBpe,
    /// SentencePiece BPE with prefix token (e.g. LLaMA / TinyStories).
    SentencePieceBpe,
    /// WordPiece tokenization (e.g. BERT / TinyBERT).
    WordPiece,
}

/// A unified Hugging Face tokenizer capable of encoding prompts and decoding generated tokens.
#[derive(Debug, Clone)]
pub struct HfTokenizer {
    pub kind: TokenizerKind,
    pub vocab: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub merges: HashMap<(String, String), usize>,
    pub special_tokens: HashMap<String, usize>,
    pub byte_to_char: [char; 256],
    pub char_to_byte: HashMap<char, u8>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub unk_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
    pub cls_token_id: Option<usize>,
    pub sep_token_id: Option<usize>,
}

fn build_gpt2_byte_char_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs = Vec::with_capacity(256);
    let mut cs = Vec::with_capacity(256);

    for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        bs.push(b);
        cs.push(b as u32);
    }

    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);

    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).unwrap_or('?');
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b);
    }

    (byte_to_char, char_to_byte)
}

impl HfTokenizer {
    /// Loads a tokenizer directly from a Hugging Face `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            EngineError::SerializationError(format!(
                "Failed to open tokenizer.json at {:?}: {}",
                path.as_ref(),
                e
            ))
        })?;
        let reader = BufReader::new(file);
        let root: serde_json::Value = serde_json::from_reader(reader).map_err(|e| {
            EngineError::SerializationError(format!("Failed to parse tokenizer.json: {}", e))
        })?;

        let model = root.get("model").ok_or_else(|| {
            EngineError::SerializationError("Missing 'model' object in tokenizer.json".to_string())
        })?;

        let model_type = model.get("type").and_then(|t| t.as_str()).unwrap_or("BPE");

        // 1. Extract vocabulary
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                EngineError::SerializationError(
                    "Missing 'model.vocab' dictionary in tokenizer.json".to_string(),
                )
            })?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0;
        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as usize;
                vocab.insert(token.clone(), id);
                if id > max_id {
                    max_id = id;
                }
            }
        }

        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            id_to_token[id] = token.clone();
        }

        // 2. Extract merges (for BPE)
        let mut merges = HashMap::new();
        if let Some(merges_arr) = model.get("merges").and_then(|m| m.as_array()) {
            for (rank, merge_val) in merges_arr.iter().enumerate() {
                if let Some(merge_str) = merge_val.as_str() {
                    let parts: Vec<&str> = merge_str.split_whitespace().collect();
                    if parts.len() == 2 {
                        merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
                    }
                } else if let Some(parts) = merge_val.as_array() {
                    if parts.len() == 2 {
                        if let (Some(a), Some(b)) = (parts[0].as_str(), parts[1].as_str()) {
                            merges.insert((a.to_string(), b.to_string()), rank);
                        }
                    }
                }
            }
        }

        // 3. Extract special tokens
        let mut special_tokens = HashMap::new();
        if let Some(added_tokens) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for item in added_tokens {
                if let (Some(id), Some(content)) = (
                    item.get("id").and_then(|i| i.as_u64()),
                    item.get("content").and_then(|c| c.as_str()),
                ) {
                    let id = id as usize;
                    special_tokens.insert(content.to_string(), id);
                    if id >= id_to_token.len() {
                        id_to_token.resize(id + 1, String::new());
                    }
                    id_to_token[id] = content.to_string();
                    vocab.insert(content.to_string(), id);
                }
            }
        }

        // 4. Identify Tokenizer kind reliably using pre_tokenizer configuration
        let pre_type = root
            .get("pre_tokenizer")
            .and_then(|p| p.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let kind = if model_type == "WordPiece" || pre_type == "BertPreTokenizer" {
            TokenizerKind::WordPiece
        } else if pre_type == "ByteLevel" || vocab.keys().filter(|k| k.contains('Ġ')).count() > 100
        {
            TokenizerKind::ByteLevelBpe
        } else {
            TokenizerKind::SentencePieceBpe
        };

        let (byte_to_char, char_to_byte) = build_gpt2_byte_char_maps();

        // 5. Detect special token IDs
        let bos_token_id = special_tokens
            .get("<s>")
            .copied()
            .or_else(|| special_tokens.get("<|startoftranscript|>").copied())
            .or_else(|| special_tokens.get("<|endoftext|>").copied());

        let eos_token_id = special_tokens
            .get("</s>")
            .copied()
            .or_else(|| special_tokens.get("<|endoftext|>").copied())
            .or_else(|| special_tokens.get("<|endoftranscript|>").copied());

        let unk_token_id = special_tokens
            .get("<unk>")
            .copied()
            .or_else(|| special_tokens.get("[UNK]").copied());

        let pad_token_id = special_tokens
            .get("<pad>")
            .copied()
            .or_else(|| special_tokens.get("[PAD]").copied());

        let cls_token_id = special_tokens
            .get("[CLS]")
            .copied()
            .or_else(|| special_tokens.get("<s>").copied());

        let sep_token_id = special_tokens
            .get("[SEP]")
            .copied()
            .or_else(|| special_tokens.get("</s>").copied());

        Ok(Self {
            kind,
            vocab,
            id_to_token,
            merges,
            special_tokens,
            byte_to_char,
            char_to_byte,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id,
            cls_token_id,
            sep_token_id,
        })
    }

    /// Encodes an input string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.kind {
            TokenizerKind::ByteLevelBpe => self.encode_gpt2_bpe(text),
            TokenizerKind::SentencePieceBpe => self.encode_spm_bpe(text),
            TokenizerKind::WordPiece => self.encode_wordpiece(text),
        }
    }

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        match self.kind {
            TokenizerKind::ByteLevelBpe => {
                let mut bytes = Vec::new();
                for &id in tokens {
                    if let Some(token_str) = self.id_to_token.get(id) {
                        for ch in token_str.chars() {
                            if let Some(&b) = self.char_to_byte.get(&ch) {
                                bytes.push(b);
                            } else {
                                bytes.extend(ch.to_string().into_bytes());
                            }
                        }
                    }
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
            TokenizerKind::SentencePieceBpe => {
                let mut out = String::new();
                for &id in tokens {
                    if let Some(token_str) = self.id_to_token.get(id) {
                        if token_str == "<s>" || token_str == "</s>" || token_str == "<unk>" {
                            continue;
                        }
                        // Handle byte tokens like <0x0A>
                        if token_str.starts_with("<0x")
                            && token_str.ends_with('>')
                            && token_str.len() == 6
                        {
                            if let Ok(b) = u8::from_str_radix(&token_str[3..5], 16) {
                                out.push(b as char);
                                continue;
                            }
                        }
                        let clean = token_str.replace('\u{2581}', " ");
                        out.push_str(&clean);
                    }
                }
                out
            }
            TokenizerKind::WordPiece => {
                let mut words = Vec::new();
                for &id in tokens {
                    if let Some(token_str) = self.id_to_token.get(id) {
                        if token_str == "[CLS]" || token_str == "[SEP]" || token_str == "[PAD]" {
                            continue;
                        }
                        if let Some(stripped) = token_str.strip_prefix("##") {
                            if let Some(last) = words.last_mut() {
                                *last = format!("{}{}", last, stripped);
                            } else {
                                words.push(stripped.to_string());
                            }
                        } else {
                            words.push(token_str.clone());
                        }
                    }
                }
                words.join(" ")
            }
        }
    }

    /// Decodes a single token ID into a UTF-8 string (ideal for real-time token streaming).
    pub fn decode_token(&self, token_id: usize) -> String {
        self.decode(&[token_id])
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Returns the BOS token ID if defined.
    pub fn bos_token_id(&self) -> Option<usize> {
        self.bos_token_id
    }

    /// Returns the EOS token ID if defined.
    pub fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id
    }

    /// Returns the PAD token ID if defined.
    pub fn pad_token_id(&self) -> Option<usize> {
        self.pad_token_id
    }

    /// Returns the UNK token ID if defined.
    pub fn unk_token_id(&self) -> Option<usize> {
        self.unk_token_id
    }

    /// Returns the CLS token ID if defined.
    pub fn cls_token_id(&self) -> Option<usize> {
        self.cls_token_id
    }

    /// Returns the SEP token ID if defined.
    pub fn sep_token_id(&self) -> Option<usize> {
        self.sep_token_id
    }

    // --- Private BPE & WordPiece Algorithms ---

    fn encode_gpt2_bpe(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        // Regex-style chunking around whitespace and words
        let chunks = split_gpt2_chunks(text);
        for chunk in chunks {
            // Map each byte to GPT-2 unicode char
            let mut pieces: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();

            self.apply_bpe_merges(&mut pieces);

            for piece in pieces {
                if let Some(&id) = self.vocab.get(&piece) {
                    tokens.push(id);
                }
            }
        }
        tokens
    }

    fn encode_spm_bpe(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let mut pieces: Vec<String> = Vec::new();
            // Prefix space via \u{2581}
            let prefixed = format!("\u{2581}{}", word);
            for ch in prefixed.chars() {
                pieces.push(ch.to_string());
            }

            self.apply_bpe_merges(&mut pieces);

            for piece in pieces {
                if let Some(&id) = self.vocab.get(&piece) {
                    tokens.push(id);
                } else if let Some(unk) = self.unk_token_id {
                    tokens.push(unk);
                }
            }
            let _ = i;
        }
        tokens
    }

    fn encode_wordpiece(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        if let Some(cls) = self.cls_token_id {
            tokens.push(cls);
        }

        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for word in words {
            let mut start = 0;
            let mut is_bad = false;
            let mut sub_tokens = Vec::new();

            while start < word.len() {
                let mut end = word.len();
                let mut cur_substr = None;

                while start < end {
                    let substr = &word[start..end];
                    let candidate = if start > 0 {
                        format!("##{}", substr)
                    } else {
                        substr.to_string()
                    };

                    if self.vocab.contains_key(&candidate) {
                        cur_substr = Some(candidate);
                        break;
                    }
                    end -= 1;
                }

                if let Some(matched) = cur_substr {
                    sub_tokens.push(self.vocab[&matched]);
                    start = end;
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                if let Some(unk) = self.unk_token_id {
                    tokens.push(unk);
                }
            } else {
                tokens.extend(sub_tokens);
            }
        }

        if let Some(sep) = self.sep_token_id {
            tokens.push(sep);
        }

        tokens
    }

    fn apply_bpe_merges(&self, pieces: &mut Vec<String>) {
        if pieces.len() < 2 {
            return;
        }

        loop {
            let mut best_pair = None;
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;

            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                        best_idx = i;
                    }
                }
            }

            if let Some(pair) = best_pair {
                let merged = format!("{}{}", pair.0, pair.1);
                pieces[best_idx] = merged;
                pieces.remove(best_idx + 1);
            } else {
                break;
            }
        }
    }
}

/// Splits text into GPT-2 word chunks handling leading whitespace.
fn split_gpt2_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
            }
            current.push(ch);
        } else if ch.is_alphanumeric() {
            current.push(ch);
        } else {
            // Punctuation
            if !current.is_empty() && !current.ends_with(' ') {
                chunks.push(current);
                current = String::new();
            }
            current.push(ch);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_byte_map_roundtrip() {
        let (byte_to_char, char_to_byte) = build_gpt2_byte_char_maps();
        for b in 0..=255u8 {
            let ch = byte_to_char[b as usize];
            assert_eq!(char_to_byte.get(&ch), Some(&b));
        }
        assert_eq!(byte_to_char[b' ' as usize], 'Ġ');
        assert_eq!(byte_to_char[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn test_gpt2_tokenizer_json_load_and_decode() {
        let path = Path::new("checkpoints/gpt2/tokenizer.json");
        if !path.exists() {
            return;
        }

        let tokenizer = HfTokenizer::from_file(path).expect("Failed to load gpt2 tokenizer.json");
        assert_eq!(tokenizer.vocab_size(), 50257);

        let decoded = tokenizer.decode(&[15496, 995]);
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_tinyllamas_tokenizer_json_load_and_decode() {
        let path = Path::new("checkpoints/tinyllamas/tokenizer.json");
        if !path.exists() {
            return;
        }

        let tokenizer =
            HfTokenizer::from_file(path).expect("Failed to load tinyllamas tokenizer.json");
        let decoded = tokenizer.decode(&[1, 9038, 2501, 263, 931]);
        assert_eq!(decoded.trim(), "Once upon a time");
    }

    #[test]
    fn test_tinybert_tokenizer_json_load_and_decode() {
        let path = Path::new("checkpoints/tinybert/tokenizer.json");
        if !path.exists() {
            return;
        }

        let tokenizer =
            HfTokenizer::from_file(path).expect("Failed to load tinybert tokenizer.json");
        assert_eq!(tokenizer.kind, TokenizerKind::WordPiece);
        let decoded = tokenizer.decode(&[101, 7592, 2088, 102]);
        assert_eq!(decoded.trim(), "hello world");
    }

    #[test]
    fn test_whisper_tokenizer_json_load_and_decode() {
        let path = Path::new("checkpoints/whisper/tokenizer.json");
        if !path.exists() {
            return;
        }

        let tokenizer =
            HfTokenizer::from_file(path).expect("Failed to load whisper tokenizer.json");
        assert_eq!(tokenizer.kind, TokenizerKind::ByteLevelBpe);
        assert_eq!(tokenizer.vocab_size(), 51865);
    }
}
