# Pretrained Model Catalog (`models`)

Reference implementations of standard foundation models and vision backbones with HuggingFace checkpoint compatibility.

**Source files**:
- Module Root: [`src/models/mod.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/models/mod.rs)
- Weight Conversion Script: [`scripts/convert_hf_model.py`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/scripts/convert_hf_model.py)

---

## Architectural Comparison Matrix

| Model | Domain / Task | Config Struct | Preset Constructors | Key Distinguishing Features |
|---|---|---|---|---|
| [**LLaMA-2**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/llama.md) | Causal Language Modeling | `LlamaConfig` | `llama_7b()`, `llama_tiny()` | RMSNorm, RoPE embeddings, SwiGLU FFN, Grouped-Query Attention (GQA) |
| [**GPT-2**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/gpt2.md) | Autoregressive Generation | `GPT2Config` | `gpt2_124m()`, `gpt2_medium()` | Pre-LN Transformer, learned position embeddings, Conv1D weight transposition |
| [**BERT**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/bert.md) | QA & Dense Embeddings | `BertConfig` | `all_minilm_l6_v2()`, `tinybert()` | Bidirectional encoder, WordPiece tokenizer, `BertForQuestionAnswering` span heads |
| [**ModernBERT**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/modern-bert.md) | Sequence Embedding / Classifier | `ModernBertConfig` | `modern_bert_base()` | Alternating local sliding window & global attention, GeGLU activations, unpadded sequences |
| [**Vision Transformer**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/vit.md) | Image Classification | `ViTConfig` | `vit_tiny_patch16_224()` | Patch projection stem ($16 \times 16$), learnable `[CLS]` token, 1D position embeddings |
| [**ResNet**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/resnet.md) | Vision Backbone | `ResNet` | `resnet18()`, `resnet34()`, `resnet50()` | BasicBlock ($3 \times 3$) and Bottleneck ($1 \times 1 \to 3 \times 3 \to 1 \times 1$) residual skips |
| [**Whisper**](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/whisper.md) | Speech Recognition (ASR) | `WhisperConfig` | `whisper_tiny()` | Conv1D acoustic stem, cross-attention encoder-decoder, 80-channel log-mel frontend |

---

## Loading HuggingFace Checkpoints

Models implement `.load_safetensors(&path)` directly from standard HuggingFace `.safetensors` files.

### Converting Weights with `scripts/convert_hf_model.py`
To download and convert any HuggingFace model into local safetensors format:
```bash
python3 scripts/convert_hf_model.py \
    --model-id "sentence-transformers/all-MiniLM-L6-v2" \
    --output-dir checkpoints/minilm
```

This exports `model.safetensors` alongside `tokenizer.json` directly into the target directory for seamless loading in Rust.

---

## Standalone CLI Integration

| CLI Binary | Underlying Engine Model | Primary Function |
|---|---|---|
| `generate` | `GPT2Model` / `Llama2LM` | Autoregressive streaming text generation with KV-Cache & sampling |
| `similarity` | `BertModel` (`all-MiniLM-L6-v2`) / `ModernBertModel` | Dense sentence embeddings, mean pooling, cosine similarity ranking |
| `qa` | `BertForQuestionAnswering` (`dynamic_tinybert`) + `all-MiniLM-L6-v2` | Extractive Question Answering with dense chunk retrieval (RAG) |

---

## See Also
- [Command-Line Interfaces](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/cli.md)
- [Serialization & SafeTensors](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/serialization.md)
- [Tokenizers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/tokenizers.md)
