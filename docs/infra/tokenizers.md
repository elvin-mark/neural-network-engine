# Tokenizers (`tokenizer`)

Subword tokenization engines: pure-Rust Byte-Level Byte-Pair Encoding (`ByteLevelBPE`) and native parser for HuggingFace `tokenizer.json` files (`HfTokenizer`).

**Source files**:
- Byte-Level BPE: [`src/tokenizer/bpe.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/tokenizer/bpe.rs)
- HuggingFace Tokenizer: [`src/tokenizer/hf_tokenizer.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/tokenizer/hf_tokenizer.rs)
- Unit tests: [`tests/tokenizer_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/tokenizer_tests.rs), [`tests/hf_parity_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/hf_parity_tests.rs)
- Example: [`examples/11_llama_bpe_training.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/examples/11_llama_bpe_training.rs)

---

## Comparison of Tokenizer Engines

| Feature | `ByteLevelBPE` | `HfTokenizer` |
|---|---|---|
| **Primary Use** | Training custom BPE tokenizers from scratch, GPT-2 tokenization | Loading pre-existing HuggingFace `tokenizer.json` files |
| **Model Types** | Byte-Level BPE (GPT-2, LLaMA) | WordPiece (BERT), BPE, Unigram |
| **Pre-tokenizers** | Byte fallback, regex pre-tokenization | Full HF pipeline (Whitespace, Metaspace, ByteLevel, etc.) |
| **Special Tokens** | Configurable (`<pad>`, `<unk>`, `<s>`, `</s>`) | Full extraction from `tokenizer.json` metadata |

---

## 1. `ByteLevelBPE` (Trainable In-Engine)

Implements byte-to-unicode character mapping that maps every raw byte ($0 \dots 255$) to a printable unicode character, preventing out-of-vocabulary (`[UNK]`) errors across arbitrary binary text.

### Training from Raw Text
```rust
let mut bpe = ByteLevelBPE::new();
bpe.train(&corpus_texts, vocab_size: 4000, min_frequency: 2)?;
bpe.save("custom_tokenizer.json")?;
```

---

## 2. `HfTokenizer` (HuggingFace Format Parser)

Parses standard HuggingFace `tokenizer.json` files without external Rust crates or dependencies.

### Capabilities:
- **Normalizers**: Lowercase, StripAccents, BertNormalizer.
- **Pre-tokenizers**: WhitespaceSplit, ByteLevel, BertPreTokenizer.
- **Token Models**:
  - **WordPiece**: Greedily matches maximal prefixes with `##` continuation markers (used by TinyBERT, MiniLM).
  - **BPE**: Merges frequent byte/character pairs.
- **Special Token Extraction**: Automatically queries `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, and `[MASK]` token IDs.

```rust
let tokenizer = HfTokenizer::from_file("checkpoints/tinybert/tokenizer.json")?;

let tokens = tokenizer.encode("What is the capital of France?");
let text = tokenizer.decode(&tokens);
```

---

## See Also
- [Text Generation CLI](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/cli.md)
- [Question Answering CLI](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/cli.md)
- [BERT Architecture](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/bert.md)
