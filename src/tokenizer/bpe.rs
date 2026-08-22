//! Pure-Rust Byte-Level Byte-Pair Encoding (BPE) Tokenizer implementation.
//!
//! Provides a byte-level BPE tokenizer with 100% UTF-8 coverage (no out-of-vocabulary errors),
//! word/whitespace pre-tokenization boundary preservation, configurable special tokens,
//! and JSON serialization.

use crate::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Serializable schema for saving and loading BPE tokenizers to/from JSON.
#[derive(Serialize, Deserialize)]
struct SerializableBPE {
    special_tokens: Vec<(String, usize)>,
    vocab_bytes: Vec<Vec<u8>>,
    merges: Vec<(usize, usize)>,
}

/// A Byte-Level Byte-Pair Encoding (BPE) Tokenizer.
///
/// Ensures 100% UTF-8 coverage by using all 256 raw bytes as the base vocabulary,
/// combined with learned byte-pair merge rules and optional special tokens (e.g. `<s>`, `</s>`, `<pad>`).
#[derive(Debug, Clone, PartialEq)]
pub struct ByteLevelBPE {
    /// Maps token ID -> Byte sequence.
    id_to_bytes: Vec<Vec<u8>>,
    /// Maps byte sequence -> Token ID.
    bytes_to_id: HashMap<Vec<u8>, usize>,
    /// Special tokens mapping: name (e.g. `"<s>"`) -> Token ID.
    special_tokens: HashMap<String, usize>,
    /// Inverse special tokens mapping: Token ID -> name.
    id_to_special: HashMap<usize, String>,
    /// Ordered list of merge pairs `(id_a, id_b)` in order of creation.
    merges: Vec<(usize, usize)>,
    /// Merge priority lookup: `(id_a, id_b)` -> priority rank (lower is higher priority).
    merge_ranks: HashMap<(usize, usize), usize>,
}

impl Default for ByteLevelBPE {
    fn default() -> Self {
        Self::with_special_tokens(&["<unk>", "<s>", "</s>", "<pad>"])
    }
}

impl ByteLevelBPE {
    /// Creates a base BPE tokenizer initialized with special tokens and all 256 raw bytes.
    pub fn with_special_tokens(special_tokens: &[&str]) -> Self {
        let mut id_to_bytes = Vec::new();
        let mut bytes_to_id = HashMap::new();
        let mut special_map = HashMap::new();
        let mut id_to_special = HashMap::new();

        // 1. Assign special tokens
        for (i, &name) in special_tokens.iter().enumerate() {
            special_map.insert(name.to_string(), i);
            id_to_special.insert(i, name.to_string());
            let bytes = name.as_bytes().to_vec();
            id_to_bytes.push(bytes.clone());
            bytes_to_id.insert(bytes, i);
        }

        let special_count = special_tokens.len();

        // 2. Assign base 256 byte tokens
        for b in 0u8..=255u8 {
            let id = special_count + (b as usize);
            let bytes = vec![b];
            id_to_bytes.push(bytes.clone());
            bytes_to_id.insert(bytes, id);
        }

        Self {
            id_to_bytes,
            bytes_to_id,
            special_tokens: special_map,
            id_to_special,
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
        }
    }

    /// Splits text into pre-tokenization chunks (words, numbers, whitespace, punctuation).
    pub fn pre_tokenize(text: &str) -> Vec<&str> {
        let mut chunks = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some((start, c)) = chars.next() {
            if c.is_alphanumeric() {
                let mut end = start + c.len_utf8();
                while let Some(&(_, next_c)) = chars.peek() {
                    if next_c.is_alphanumeric() {
                        end += next_c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                chunks.push(&text[start..end]);
            } else if c.is_whitespace() {
                let mut end = start + c.len_utf8();
                while let Some(&(_, next_c)) = chars.peek() {
                    if next_c.is_whitespace() {
                        end += next_c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                chunks.push(&text[start..end]);
            } else {
                // Individual punctuation or symbol character
                let end = start + c.len_utf8();
                chunks.push(&text[start..end]);
            }
        }

        chunks
    }

    /// Maps a raw byte to its initial base token ID.
    #[inline]
    fn byte_to_id(&self, b: u8) -> usize {
        self.special_tokens.len() + (b as usize)
    }

    /// Replaces consecutive occurrences of `(a, b)` with `new_id` in a word token slice.
    fn merge_pair_in_word(word: &[usize], a: usize, b: usize, new_id: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(word.len());
        let mut i = 0;
        while i < word.len() {
            if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                result.push(new_id);
                i += 2;
            } else {
                result.push(word[i]);
                i += 1;
            }
        }
        result
    }

    /// Trains a byte-level BPE tokenizer on the given corpus up to `target_vocab_size`.
    pub fn train(corpus: &str, target_vocab_size: usize, special_tokens: &[&str]) -> Result<Self> {
        let mut tokenizer = Self::with_special_tokens(special_tokens);
        let base_vocab_size = tokenizer.id_to_bytes.len();

        if target_vocab_size <= base_vocab_size {
            return Ok(tokenizer);
        }

        // 1. Pre-tokenize corpus and count word frequencies
        let chunks = Self::pre_tokenize(corpus);
        let mut word_counts: HashMap<Vec<usize>, usize> = HashMap::new();

        for chunk in chunks {
            let token_ids: Vec<usize> = chunk
                .as_bytes()
                .iter()
                .map(|&b| tokenizer.byte_to_id(b))
                .collect();
            if !token_ids.is_empty() {
                *word_counts.entry(token_ids).or_default() += 1;
            }
        }

        // 2. Iteratively find most frequent adjacent pair and merge
        while tokenizer.id_to_bytes.len() < target_vocab_size {
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();

            for (word, count) in &word_counts {
                if word.len() < 2 {
                    continue;
                }
                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    *pair_counts.entry(pair).or_default() += count;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find pair with highest frequency (deterministic tie-breaking)
            let mut best_pair = None;
            let mut max_freq = 0;

            for (pair, &freq) in &pair_counts {
                if freq > max_freq || (freq == max_freq && best_pair.map_or(true, |p| *pair < p)) {
                    max_freq = freq;
                    best_pair = Some(*pair);
                }
            }

            let (best_a, best_b) = match best_pair {
                Some(pair) if max_freq > 0 => pair,
                _ => break,
            };

            let new_id = tokenizer.id_to_bytes.len();
            let mut merged_bytes = tokenizer.id_to_bytes[best_a].clone();
            merged_bytes.extend_from_slice(&tokenizer.id_to_bytes[best_b]);

            let rank = tokenizer.merges.len();
            tokenizer.merges.push((best_a, best_b));
            tokenizer.merge_ranks.insert((best_a, best_b), rank);

            tokenizer.bytes_to_id.insert(merged_bytes.clone(), new_id);
            tokenizer.id_to_bytes.push(merged_bytes);

            // Update word_counts with merged tokens
            let mut new_word_counts = HashMap::with_capacity(word_counts.len());
            for (word, count) in word_counts {
                let updated = Self::merge_pair_in_word(&word, best_a, best_b, new_id);
                *new_word_counts.entry(updated).or_default() += count;
            }
            word_counts = new_word_counts;
        }

        Ok(tokenizer)
    }

    /// Encodes a word chunk into token IDs using learned merge rules.
    fn encode_chunk(&self, chunk: &str) -> Vec<usize> {
        let mut tokens: Vec<usize> = chunk
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_id(b))
            .collect();

        if tokens.len() < 2 {
            return tokens;
        }

        loop {
            // Find lowest rank (highest priority) merge pair in tokens
            let mut best_merge: Option<((usize, usize), usize, usize)> = None; // ((a, b), rank, index)

            for (idx, window) in tokens.windows(2).enumerate() {
                let pair = (window[0], window[1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_merge.as_ref().map_or(true, |m| rank < m.1) {
                        best_merge = Some((pair, rank, idx));
                    }
                }
            }

            match best_merge {
                Some(((best_a, best_b), _, _)) => {
                    let new_id = self.bytes_to_id[&[
                        self.id_to_bytes[best_a].as_slice(),
                        self.id_to_bytes[best_b].as_slice(),
                    ]
                    .concat()];
                    tokens = Self::merge_pair_in_word(&tokens, best_a, best_b, new_id);
                }
                None => break,
            }
        }

        tokens
    }

    /// Encodes a text slice that does not contain special tokens.
    fn encode_normal_text(&self, text: &str) -> Vec<usize> {
        let mut result = Vec::new();
        let chunks = Self::pre_tokenize(text);
        for chunk in chunks {
            result.extend(self.encode_chunk(chunk));
        }
        result
    }

    /// Encodes text into a sequence of token IDs, recognizing special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        if self.special_tokens.is_empty() {
            return Ok(self.encode_normal_text(text));
        }

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find earliest occurrence of any special token in remaining
            let mut earliest_match: Option<(usize, usize, usize)> = None; // (start_pos, end_pos, special_id)

            for (name, &special_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(name) {
                    if earliest_match.as_ref().map_or(true, |m| pos < m.0) {
                        earliest_match = Some((pos, pos + name.len(), special_id));
                    }
                }
            }

            match earliest_match {
                Some((start, end, special_id)) => {
                    if start > 0 {
                        result.extend(self.encode_normal_text(&remaining[..start]));
                    }
                    result.push(special_id);
                    remaining = &remaining[end..];
                }
                None => {
                    result.extend(self.encode_normal_text(remaining));
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, tokens: &[usize]) -> Result<String> {
        let mut bytes = Vec::new();

        for &id in tokens {
            if id >= self.id_to_bytes.len() {
                return Err(EngineError::TokenizerError(format!(
                    "Token ID {} is out of vocabulary range (vocab size: {})",
                    id,
                    self.id_to_bytes.len()
                )));
            }
            bytes.extend_from_slice(&self.id_to_bytes[id]);
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Returns the total vocabulary size (special tokens + 256 base bytes + merged tokens).
    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Returns the token ID for a given special token name, if present.
    pub fn special_token_id(&self, name: &str) -> Option<usize> {
        self.special_tokens.get(name).copied()
    }

    /// Returns the BOS (Beginning of Sequence) token ID if `<s>` is defined.
    pub fn bos_token_id(&self) -> Option<usize> {
        self.special_token_id("<s>")
    }

    /// Returns the EOS (End of Sequence) token ID if `</s>` is defined.
    pub fn eos_token_id(&self) -> Option<usize> {
        self.special_token_id("</s>")
    }

    /// Returns the PAD token ID if `<pad>` is defined.
    pub fn pad_token_id(&self) -> Option<usize> {
        self.special_token_id("<pad>")
    }

    /// Returns the UNK token ID if `<unk>` is defined.
    pub fn unk_token_id(&self) -> Option<usize> {
        self.special_token_id("<unk>")
    }

    /// Looks up the token ID corresponding to an exact byte sequence.
    pub fn token_to_id(&self, token_bytes: &[u8]) -> Option<usize> {
        self.bytes_to_id.get(token_bytes).copied()
    }

    /// Looks up the byte slice corresponding to a token ID.
    pub fn id_to_token(&self, id: usize) -> Option<&[u8]> {
        self.id_to_bytes.get(id).map(|v| v.as_slice())
    }

    /// Saves the tokenizer vocabulary, special tokens, and merge rules to a JSON file.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            EngineError::SerializationError(format!("Failed to create tokenizer file: {}", e))
        })?;
        let writer = BufWriter::new(file);

        let serializable = SerializableBPE {
            special_tokens: self
                .special_tokens
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect(),
            vocab_bytes: self.id_to_bytes.clone(),
            merges: self.merges.clone(),
        };

        serde_json::to_writer_pretty(writer, &serializable).map_err(|e| {
            EngineError::SerializationError(format!("Failed to write tokenizer JSON: {}", e))
        })?;

        Ok(())
    }

    /// Loads a tokenizer vocabulary and merge rules from a JSON file.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            EngineError::SerializationError(format!("Failed to open tokenizer file: {}", e))
        })?;
        let reader = BufReader::new(file);

        let serializable: SerializableBPE = serde_json::from_reader(reader).map_err(|e| {
            EngineError::SerializationError(format!("Failed to parse tokenizer JSON: {}", e))
        })?;

        let mut special_tokens = HashMap::new();
        let mut id_to_special = HashMap::new();
        for (name, id) in serializable.special_tokens {
            special_tokens.insert(name.clone(), id);
            id_to_special.insert(id, name);
        }

        let mut bytes_to_id = HashMap::new();
        for (id, bytes) in serializable.vocab_bytes.iter().enumerate() {
            bytes_to_id.insert(bytes.clone(), id);
        }

        let mut merge_ranks = HashMap::new();
        for (rank, &pair) in serializable.merges.iter().enumerate() {
            merge_ranks.insert(pair, rank);
        }

        Ok(Self {
            id_to_bytes: serializable.vocab_bytes,
            bytes_to_id,
            special_tokens,
            id_to_special,
            merges: serializable.merges,
            merge_ranks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_training_and_roundtrip() {
        let text = "Once upon a time, there was a little girl named Lily. Lily loved to play in the garden. Once upon a time, Lily found a golden key in the garden.";
        let tokenizer = ByteLevelBPE::train(text, 280, &["<s>", "</s>"]).unwrap();

        assert_eq!(tokenizer.vocab_size(), 280);
        assert_eq!(tokenizer.bos_token_id(), Some(0));
        assert_eq!(tokenizer.eos_token_id(), Some(1));

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
        // Merged tokens should compress the raw byte count
        assert!(encoded.len() < text.len());
    }

    #[test]
    fn test_bpe_unicode_coverage() {
        let text =
            "Hello 🌍 World! Rust 🦀 Tokenizer with UTF-8 support: ñ, ü, こんにちは, Привет!";
        let tokenizer = ByteLevelBPE::default();

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_json_serialization_roundtrip() {
        let text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.";
        let tokenizer = ByteLevelBPE::train(text, 275, &["<unk>", "<s>", "</s>"]).unwrap();

        let temp_path = std::env::temp_dir().join("test_bpe_tokenizer.json");
        tokenizer.save_json(&temp_path).unwrap();

        let loaded = ByteLevelBPE::load_json(&temp_path).unwrap();
        assert_eq!(tokenizer, loaded);

        let encoded_orig = tokenizer.encode(text).unwrap();
        let encoded_loaded = loaded.encode(text).unwrap();
        assert_eq!(encoded_orig, encoded_loaded);

        let _ = std::fs::remove_file(temp_path);
    }
}
