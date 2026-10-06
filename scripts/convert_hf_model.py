#!/usr/bin/env python3
"""
Convert Hugging Face models and tokenizers into Neural Network Engine format.

Supported models:
- gpt2 (OpenAI GPT-2 124M)

Outputs:
- <output_dir>/model.safetensors: Transformed weights in F32 format.
- <output_dir>/tokenizer.json: Tokenizer configuration and vocabulary.
- <output_dir>/reference.json: Reference prompt, input token IDs, and generated token IDs.
"""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration
from safetensors.torch import save_file


def convert_gpt2(model_id: str, output_dir: Path):
    print(f"[*] Loading Hugging Face GPT-2 model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Transform weights to match neural-network-engine GPT2Model naming & Conv1D transpose
    print("[*] Transforming state_dict for neural-network-engine...")
    transformed_weights = {}

    # Token embeddings & position embeddings
    transformed_weights["wte.weight"] = model.transformer.wte.weight.detach().to(torch.float32).contiguous()
    transformed_weights["wpe.weight"] = model.transformer.wpe.weight.detach().to(torch.float32).contiguous()

    # Cascade through transformer decoder blocks
    num_layers = len(model.transformer.h)
    d_model = model.config.n_embd

    for i in range(num_layers):
        hf_layer = model.transformer.h[i]
        prefix = f"blocks.{i}"

        # LayerNorm 1
        transformed_weights[f"{prefix}.ln_1.weight"] = hf_layer.ln_1.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.ln_1.bias"] = hf_layer.ln_1.bias.detach().to(torch.float32).contiguous()

        # MultiHeadAttention: HF c_attn is Conv1D [d_model, 3 * d_model]
        # In our engine: q_proj, k_proj, v_proj are Linear layers [d_model, d_model] -> weight [out_features, in_features] = [d_model, d_model]
        c_attn_w = hf_layer.attn.c_attn.weight.detach().to(torch.float32)  # [d_model, 3 * d_model]
        c_attn_b = hf_layer.attn.c_attn.bias.detach().to(torch.float32)    # [3 * d_model]

        # c_attn.weight in Conv1D maps x @ W + b. For Linear (y = x @ W^T), W_linear = W_conv1d.T
        c_attn_w_t = c_attn_w.t() # [3 * d_model, d_model]
        q_w, k_w, v_w = c_attn_w_t.split(d_model, dim=0)
        q_b, k_b, v_b = c_attn_b.split(d_model, dim=0)

        transformed_weights[f"{prefix}.attn.q_proj.weight"] = q_w.contiguous()
        transformed_weights[f"{prefix}.attn.q_proj.bias"] = q_b.contiguous()
        transformed_weights[f"{prefix}.attn.k_proj.weight"] = k_w.contiguous()
        transformed_weights[f"{prefix}.attn.k_proj.bias"] = k_b.contiguous()
        transformed_weights[f"{prefix}.attn.v_proj.weight"] = v_w.contiguous()
        transformed_weights[f"{prefix}.attn.v_proj.bias"] = v_b.contiguous()

        # Attention out_proj (c_proj in HF)
        c_proj_w = hf_layer.attn.c_proj.weight.detach().to(torch.float32).t()
        c_proj_b = hf_layer.attn.c_proj.bias.detach().to(torch.float32)
        transformed_weights[f"{prefix}.attn.out_proj.weight"] = c_proj_w.contiguous()
        transformed_weights[f"{prefix}.attn.out_proj.bias"] = c_proj_b.contiguous()

        # LayerNorm 2
        transformed_weights[f"{prefix}.ln_2.weight"] = hf_layer.ln_2.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.ln_2.bias"] = hf_layer.ln_2.bias.detach().to(torch.float32).contiguous()

        # MLP: c_fc and c_proj
        mlp_fc_w = hf_layer.mlp.c_fc.weight.detach().to(torch.float32).t()
        mlp_fc_b = hf_layer.mlp.c_fc.bias.detach().to(torch.float32)
        transformed_weights[f"{prefix}.mlp_fc.weight"] = mlp_fc_w.contiguous()
        transformed_weights[f"{prefix}.mlp_fc.bias"] = mlp_fc_b.contiguous()

        mlp_proj_w = hf_layer.mlp.c_proj.weight.detach().to(torch.float32).t()
        mlp_proj_b = hf_layer.mlp.c_proj.bias.detach().to(torch.float32)
        transformed_weights[f"{prefix}.mlp_proj.weight"] = mlp_proj_w.contiguous()
        transformed_weights[f"{prefix}.mlp_proj.bias"] = mlp_proj_b.contiguous()

    # Final LayerNorm
    transformed_weights["ln_f.weight"] = model.transformer.ln_f.weight.detach().to(torch.float32).contiguous()
    transformed_weights["ln_f.bias"] = model.transformer.ln_f.bias.detach().to(torch.float32).contiguous()

    # LM Head
    transformed_weights["lm_head.weight"] = model.lm_head.weight.detach().to(torch.float32).clone().contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    # 2. Save Tokenizer
    tok_path = output_dir / "tokenizer.json"
    print(f"[*] Saving Tokenizer to {tok_path}...")
    tokenizer.save_pretrained(str(output_dir))

    # 3. Generate reference test vectors for parity verification
    print("[*] Generating reference verification sequence...")
    prompt = "Hello, my name is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids)
        logits_first_step = out.logits[:, -1, :].tolist()
        gen_ids = model.generate(input_ids, max_new_tokens=5, do_sample=False)[0].tolist()

    gen_text = tokenizer.decode(gen_ids)
    print(f"[*] Reference Prompt: {repr(prompt)}")
    print(f"[*] Reference Generated IDs: {gen_ids}")
    print(f"[*] Reference Output Text: {repr(gen_text)}")

    reference_meta = {
        "model_id": model_id,
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": gen_ids,
        "generated_text": gen_text,
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_tinyllamas(model_id: str, output_dir: Path):
    print(f"[*] Loading Hugging Face TinyStories LLaMA model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine Llama2LM...")
    transformed_weights = {}

    # Token embeddings & lm_head
    transformed_weights["tok_embeddings.weight"] = model.model.embed_tokens.weight.detach().to(torch.float32).contiguous()
    transformed_weights["lm_head.weight"] = model.lm_head.weight.detach().to(torch.float32).clone().contiguous()
    transformed_weights["norm.weight"] = model.model.norm.weight.detach().to(torch.float32).contiguous()

    # Cascade through LLaMA decoder layers
    for i, layer in enumerate(model.model.layers):
        prefix = f"layers.{i}"

        # Attention RMSNorm
        transformed_weights[f"{prefix}.attn_norm.weight"] = layer.input_layernorm.weight.detach().to(torch.float32).contiguous()

        # Attention projections (q, k, v, o)
        transformed_weights[f"{prefix}.attn.q_proj.weight"] = layer.self_attn.q_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.k_proj.weight"] = layer.self_attn.k_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.v_proj.weight"] = layer.self_attn.v_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.o_proj.weight"] = layer.self_attn.o_proj.weight.detach().to(torch.float32).contiguous()

        # FFN RMSNorm
        transformed_weights[f"{prefix}.ffn_norm.weight"] = layer.post_attention_layernorm.weight.detach().to(torch.float32).contiguous()

        # SwiGLU MLP: gate_proj, up_proj, down_proj
        transformed_weights[f"{prefix}.ffn.gate_proj.weight"] = layer.mlp.gate_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.ffn.up_proj.weight"] = layer.mlp.up_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.ffn.down_proj.weight"] = layer.mlp.down_proj.weight.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    tok_path = output_dir / "tokenizer.json"
    print(f"[*] Saving Tokenizer to {tok_path}...")
    tokenizer.save_pretrained(str(output_dir))

    # Reference prompt and generation
    print("[*] Generating reference verification sequence...")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        gen_ids = model.generate(input_ids, max_new_tokens=5, do_sample=False)[0].tolist()

    gen_text = tokenizer.decode(gen_ids)
    print(f"[*] Reference Prompt: {repr(prompt)}")
    print(f"[*] Reference Generated IDs: {gen_ids}")
    print(f"[*] Reference Output Text: {repr(gen_text)}")

    reference_meta = {
        "model_id": model_id,
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": gen_ids,
        "generated_text": gen_text,
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_bert(model_id: str, output_dir: Path):
    print(f"[*] Loading Hugging Face BERT model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine BertModel...")
    transformed_weights = {}

    # Embeddings
    transformed_weights["embeddings.word_embeddings.weight"] = model.embeddings.word_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.position_embeddings.weight"] = model.embeddings.position_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.token_type_embeddings.weight"] = model.embeddings.token_type_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.layer_norm.weight"] = model.embeddings.LayerNorm.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.layer_norm.bias"] = model.embeddings.LayerNorm.bias.detach().to(torch.float32).contiguous()

    # Encoder layers
    for i, layer in enumerate(model.encoder.layer):
        prefix = f"encoder.layers.{i}"

        # Self-attention projections (q, k, v, out)
        transformed_weights[f"{prefix}.attention.q_proj.weight"] = layer.attention.self.query.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.q_proj.bias"] = layer.attention.self.query.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.k_proj.weight"] = layer.attention.self.key.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.k_proj.bias"] = layer.attention.self.key.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.v_proj.weight"] = layer.attention.self.value.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.v_proj.bias"] = layer.attention.self.value.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.attention.out_proj.weight"] = layer.attention.output.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.out_proj.bias"] = layer.attention.output.dense.bias.detach().to(torch.float32).contiguous()

        # Attention LayerNorm
        transformed_weights[f"{prefix}.attention_norm.weight"] = layer.attention.output.LayerNorm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention_norm.bias"] = layer.attention.output.LayerNorm.bias.detach().to(torch.float32).contiguous()

        # Intermediate FFN
        transformed_weights[f"{prefix}.intermediate.weight"] = layer.intermediate.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.intermediate.bias"] = layer.intermediate.dense.bias.detach().to(torch.float32).contiguous()

        # Output FFN
        transformed_weights[f"{prefix}.output_dense.weight"] = layer.output.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.output_dense.bias"] = layer.output.dense.bias.detach().to(torch.float32).contiguous()

        # Output LayerNorm
        transformed_weights[f"{prefix}.output_norm.weight"] = layer.output.LayerNorm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.output_norm.bias"] = layer.output.LayerNorm.bias.detach().to(torch.float32).contiguous()

    # Pooler
    if hasattr(model, "pooler") and model.pooler is not None:
        transformed_weights["pooler.dense.weight"] = model.pooler.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights["pooler.dense.bias"] = model.pooler.dense.bias.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    tok_path = output_dir / "tokenizer.json"
    print(f"[*] Saving Tokenizer to {tok_path}...")
    tokenizer.save_pretrained(str(output_dir))

    # Reference inference
    print("[*] Generating reference verification sequence...")
    prompt = "Hello world, BERT embeddings!"
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
        seq_output = out.last_hidden_state[0].tolist()
        pooled_output = out.pooler_output[0].tolist() if getattr(out, "pooler_output", None) is not None else []

    reference_meta = {
        "model_id": model_id,
        "prompt": prompt,
        "input_ids": enc.input_ids[0].tolist(),
        "token_type_ids": enc.token_type_ids[0].tolist() if "token_type_ids" in enc else [0] * len(enc.input_ids[0]),
        "first_5_seq_output": seq_output[0][:5],
        "first_5_pooled_output": pooled_output[:5] if pooled_output else [],
        "pooled_output": pooled_output,
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_tinybert_qa(model_id: str, output_dir: Path):
    from huggingface_hub import hf_hub_download
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertForQuestionAnswering

    print(f"[*] Loading Hugging Face TinyBERT QA model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = BertConfig.from_pretrained(model_id)
    hf_model = BertForQuestionAnswering(config)

    bin_path = hf_hub_download(model_id, "pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu", weights_only=False)
    hf_model.load_state_dict(sd, strict=False)
    hf_model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine BertForQuestionAnswering...")
    transformed_weights = {}

    # Embeddings
    transformed_weights["embeddings.word_embeddings.weight"] = hf_model.bert.embeddings.word_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.position_embeddings.weight"] = hf_model.bert.embeddings.position_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.token_type_embeddings.weight"] = hf_model.bert.embeddings.token_type_embeddings.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.layer_norm.weight"] = hf_model.bert.embeddings.LayerNorm.weight.detach().to(torch.float32).contiguous()
    transformed_weights["embeddings.layer_norm.bias"] = hf_model.bert.embeddings.LayerNorm.bias.detach().to(torch.float32).contiguous()

    # Encoder layers
    for i, layer in enumerate(hf_model.bert.encoder.layer):
        prefix = f"encoder.layers.{i}"

        transformed_weights[f"{prefix}.attention.q_proj.weight"] = layer.attention.self.query.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.q_proj.bias"] = layer.attention.self.query.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.k_proj.weight"] = layer.attention.self.key.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.k_proj.bias"] = layer.attention.self.key.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.v_proj.weight"] = layer.attention.self.value.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.v_proj.bias"] = layer.attention.self.value.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.attention.out_proj.weight"] = layer.attention.output.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention.out_proj.bias"] = layer.attention.output.dense.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.attention_norm.weight"] = layer.attention.output.LayerNorm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attention_norm.bias"] = layer.attention.output.LayerNorm.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.intermediate.weight"] = layer.intermediate.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.intermediate.bias"] = layer.intermediate.dense.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.output_dense.weight"] = layer.output.dense.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.output_dense.bias"] = layer.output.dense.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.output_norm.weight"] = layer.output.LayerNorm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.output_norm.bias"] = layer.output.LayerNorm.bias.detach().to(torch.float32).contiguous()

    # Question Answering classification head
    transformed_weights["qa_outputs.weight"] = hf_model.qa_outputs.weight.detach().to(torch.float32).contiguous()
    transformed_weights["qa_outputs.bias"] = hf_model.qa_outputs.bias.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    tok_path = output_dir / "tokenizer.json"
    print(f"[*] Saving Tokenizer to {tok_path}...")
    tokenizer.save_pretrained(str(output_dir))

    # Reference inference
    print("[*] Generating reference verification sequence...")
    question = "Where is the capital of France?"
    context = "Paris is the capital of France."
    enc = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        out = hf_model(**enc)
        start_logits = out.start_logits[0].tolist()
        end_logits = out.end_logits[0].tolist()

    best_start = int(torch.tensor(start_logits).argmax().item())
    best_end = int(torch.tensor(end_logits).argmax().item())
    answer_tokens = enc.input_ids[0][best_start : best_end + 1]
    predicted_answer = tokenizer.decode(answer_tokens).strip()

    print(f"[*] Question: {repr(question)}")
    print(f"[*] Context: {repr(context)}")
    print(f"[*] Predicted Answer: {repr(predicted_answer)}")
    print(f"[*] Start Token Index: {best_start}, End Token Index: {best_end}")

    reference_meta = {
        "model_id": model_id,
        "question": question,
        "context": context,
        "predicted_answer": predicted_answer,
        "input_ids": enc.input_ids[0].tolist(),
        "token_type_ids": enc.token_type_ids[0].tolist(),
        "best_start": best_start,
        "best_end": best_end,
        "first_5_start_logits": start_logits[:5],
        "first_5_end_logits": end_logits[:5],
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_whisper(model_id: str, output_dir: Path):
    print(f"[*] Loading Hugging Face Whisper model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine Whisper...")
    transformed_weights = {}

    # Encoder stem: HF Conv1d [C_out, C_in, kernel_size] -> Conv2d [C_out, C_in, 1, kernel_size]
    transformed_weights["encoder.conv1.weight"] = model.model.encoder.conv1.weight.unsqueeze(2).detach().to(torch.float32).contiguous()
    transformed_weights["encoder.conv1.bias"] = model.model.encoder.conv1.bias.detach().to(torch.float32).contiguous()
    transformed_weights["encoder.conv2.weight"] = model.model.encoder.conv2.weight.unsqueeze(2).detach().to(torch.float32).contiguous()
    transformed_weights["encoder.conv2.bias"] = model.model.encoder.conv2.bias.detach().to(torch.float32).contiguous()

    # Encoder position embeddings & layer norm
    transformed_weights["encoder.embed_positions.weight"] = model.model.encoder.embed_positions.weight.unsqueeze(0).detach().to(torch.float32).contiguous()
    transformed_weights["encoder.layer_norm.weight"] = model.model.encoder.layer_norm.weight.detach().to(torch.float32).contiguous()
    transformed_weights["encoder.layer_norm.bias"] = model.model.encoder.layer_norm.bias.detach().to(torch.float32).contiguous()

    # Encoder layers
    for i, layer in enumerate(model.model.encoder.layers):
        prefix = f"encoder.layers.{i}"
        transformed_weights[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.self_attn_layer_norm.weight"] = layer.self_attn_layer_norm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn_layer_norm.bias"] = layer.self_attn_layer_norm.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.fc1.weight"] = layer.fc1.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc1.bias"] = layer.fc1.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc2.weight"] = layer.fc2.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc2.bias"] = layer.fc2.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.detach().to(torch.float32).contiguous()

    # Decoder embeddings & layer norm
    transformed_weights["decoder.embed_tokens.weight"] = model.model.decoder.embed_tokens.weight.detach().to(torch.float32).contiguous()
    transformed_weights["decoder.embed_positions.weight"] = model.model.decoder.embed_positions.weight.unsqueeze(0).detach().to(torch.float32).contiguous()
    transformed_weights["decoder.layer_norm.weight"] = model.model.decoder.layer_norm.weight.detach().to(torch.float32).contiguous()
    transformed_weights["decoder.layer_norm.bias"] = model.model.decoder.layer_norm.bias.detach().to(torch.float32).contiguous()
    transformed_weights["decoder.lm_head.weight"] = model.proj_out.weight.detach().to(torch.float32).clone().contiguous()

    # Decoder layers
    for i, layer in enumerate(model.model.decoder.layers):
        prefix = f"decoder.layers.{i}"
        transformed_weights[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.self_attn_layer_norm.weight"] = layer.self_attn_layer_norm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.self_attn_layer_norm.bias"] = layer.self_attn_layer_norm.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.encoder_attn.q_proj.weight"] = layer.encoder_attn.q_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.q_proj.bias"] = layer.encoder_attn.q_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.k_proj.weight"] = layer.encoder_attn.k_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.v_proj.weight"] = layer.encoder_attn.v_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.v_proj.bias"] = layer.encoder_attn.v_proj.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.out_proj.weight"] = layer.encoder_attn.out_proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn.out_proj.bias"] = layer.encoder_attn.out_proj.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.encoder_attn_layer_norm.weight"] = layer.encoder_attn_layer_norm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.encoder_attn_layer_norm.bias"] = layer.encoder_attn_layer_norm.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.fc1.weight"] = layer.fc1.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc1.bias"] = layer.fc1.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc2.weight"] = layer.fc2.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.fc2.bias"] = layer.fc2.bias.detach().to(torch.float32).contiguous()

        transformed_weights[f"{prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    tok_path = output_dir / "tokenizer.json"
    print(f"[*] Saving Tokenizer to {tok_path}...")
    tokenizer.save_pretrained(str(output_dir))

    # Reference inference verification
    print("[*] Generating reference verification sequence...")
    # Whisper expects 3000 mel frames downsampled to 1500 encoder positions
    mel = torch.zeros(1, 80, 3000)
    # Standard Whisper decoder prompt: <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>
    decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]])

    with torch.no_grad():
        out = model(input_features=mel, decoder_input_ids=decoder_input_ids)
        logits = out.logits[0, -1, :].tolist()
        predicted_token = int(torch.tensor(logits).argmax().item())

    print(f"[*] Decoder Prompt Tokens: {decoder_input_ids[0].tolist()}")
    print(f"[*] Next Predicted Token: {predicted_token}")
    print(f"[*] First 5 Logits: {logits[:5]}")

    reference_meta = {
        "model_id": model_id,
        "decoder_input_ids": decoder_input_ids[0].tolist(),
        "next_predicted_token": predicted_token,
        "first_5_logits": logits[:5],
        "mel_shape": [1, 80, 3000],
        "mel_type": "zeros",
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_vit(model_id: str, output_dir: Path):
    print(f"[*] Loading timm ViT model '{model_id}'...")
    import timm

    timm_name = f"hf-hub:{model_id}" if "/" in model_id and not model_id.startswith(("hf-hub:", "local-dir:")) else model_id
    model = timm.create_model(timm_name, pretrained=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine VisionTransformer...")
    transformed_weights = {}

    # Patch embedding
    transformed_weights["patch_embed.proj.weight"] = model.patch_embed.proj.weight.detach().to(torch.float32).contiguous()
    transformed_weights["patch_embed.proj.bias"] = model.patch_embed.proj.bias.detach().to(torch.float32).contiguous()

    # CLS token & Positional embeddings
    transformed_weights["cls_token"] = model.cls_token.detach().to(torch.float32).contiguous()
    transformed_weights["pos_embed"] = model.pos_embed.detach().to(torch.float32).contiguous()

    # 12 Transformer blocks
    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        # LayerNorm 1
        transformed_weights[f"{prefix}.norm1.weight"] = block.norm1.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.norm1.bias"] = block.norm1.bias.detach().to(torch.float32).contiguous()

        # MultiHeadAttention: qkv chunking
        w_q, w_k, w_v = block.attn.qkv.weight.chunk(3, dim=0)
        b_q, b_k, b_v = block.attn.qkv.bias.chunk(3, dim=0)

        transformed_weights[f"{prefix}.attn.q_proj.weight"] = w_q.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.q_proj.bias"] = b_q.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.k_proj.weight"] = w_k.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.k_proj.bias"] = b_k.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.v_proj.weight"] = w_v.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.v_proj.bias"] = b_v.detach().to(torch.float32).contiguous()

        # Attention out_proj
        transformed_weights[f"{prefix}.attn.out_proj.weight"] = block.attn.proj.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.attn.out_proj.bias"] = block.attn.proj.bias.detach().to(torch.float32).contiguous()

        # LayerNorm 2
        transformed_weights[f"{prefix}.norm2.weight"] = block.norm2.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.norm2.bias"] = block.norm2.bias.detach().to(torch.float32).contiguous()

        # MLP
        transformed_weights[f"{prefix}.mlp_fc1.weight"] = block.mlp.fc1.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.mlp_fc1.bias"] = block.mlp.fc1.bias.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.mlp_fc2.weight"] = block.mlp.fc2.weight.detach().to(torch.float32).contiguous()
        transformed_weights[f"{prefix}.mlp_fc2.bias"] = block.mlp.fc2.bias.detach().to(torch.float32).contiguous()

    # Final LayerNorm & Head
    transformed_weights["norm.weight"] = model.norm.weight.detach().to(torch.float32).contiguous()
    transformed_weights["norm.bias"] = model.norm.bias.detach().to(torch.float32).contiguous()
    transformed_weights["head.weight"] = model.head.weight.detach().to(torch.float32).contiguous()
    transformed_weights["head.bias"] = model.head.bias.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    # Reference inference verification
    print("[*] Generating reference verification vector...")
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
        logits = out[0].tolist()
        predicted_class = int(out[0].argmax().item())

    print(f"[*] Predicted Class: {predicted_class}")
    print(f"[*] First 5 Logits: {logits[:5]}")

    reference_meta = {
        "model_id": model_id,
        "input_shape": [1, 3, 224, 224],
        "input_type": "zeros",
        "predicted_class": predicted_class,
        "first_5_logits": logits[:5],
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def convert_modern_bert(model_id: str, output_dir: Path):
    print(f"[*] Loading Hugging Face ModernBERT model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Transforming state_dict for neural-network-engine ModernBertModel...")
    transformed_weights = {}

    # Token embeddings
    transformed_weights["embeddings.weight"] = model.embeddings.tok_embeddings.weight.detach().to(torch.float32).contiguous()

    # Cascade through transformer layers
    for i, layer in enumerate(model.layers):
        prefix = f"encoder.layers.{i}"

        # Attn norm
        transformed_weights[f"{prefix}.attn_norm.weight"] = layer.attn_norm.weight.detach().to(torch.float32).contiguous()

        # Attention: Wqkv [3 * d_model, d_model] -> q, k, v
        wqkv = layer.attn.Wqkv.weight.detach().to(torch.float32)
        d_model = model.config.hidden_size
        q_w, k_w, v_w = wqkv.split(d_model, dim=0)

        transformed_weights[f"{prefix}.attention.q_proj.weight"] = q_w.contiguous()
        transformed_weights[f"{prefix}.attention.k_proj.weight"] = k_w.contiguous()
        transformed_weights[f"{prefix}.attention.v_proj.weight"] = v_w.contiguous()

        transformed_weights[f"{prefix}.attention.out_proj.weight"] = layer.attn.Wo.weight.detach().to(torch.float32).contiguous()

        # MLP norm
        transformed_weights[f"{prefix}.mlp_norm.weight"] = layer.mlp_norm.weight.detach().to(torch.float32).contiguous()

        # MLP: Wi [2 * intermediate_size, d_model] -> gate_proj, up_proj
        wi = layer.mlp.Wi.weight.detach().to(torch.float32)
        intermediate_size = model.config.intermediate_size
        gate_w, up_w = wi.split(intermediate_size, dim=0)

        transformed_weights[f"{prefix}.mlp.gate_proj.weight"] = gate_w.contiguous()
        transformed_weights[f"{prefix}.mlp.up_proj.weight"] = up_w.contiguous()
        transformed_weights[f"{prefix}.mlp.down_proj.weight"] = layer.mlp.Wo.weight.detach().to(torch.float32).contiguous()

    # Final norm
    transformed_weights["encoder.final_norm.weight"] = model.final_norm.weight.detach().to(torch.float32).contiguous()

    weights_path = output_dir / "model.safetensors"
    print(f"[*] Saving SafeTensors weights to {weights_path}...")
    save_file(transformed_weights, str(weights_path))

    print("[*] Saving tokenizer configuration...")
    tokenizer.save_pretrained(str(output_dir))

    text = "ModernBERT is a modernized bidirectional encoder."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        pooled = out.last_hidden_state.mean(dim=1)[0].tolist()

    reference_meta = {
        "model_id": model_id,
        "text": text,
        "input_ids": inputs.input_ids[0].tolist(),
        "first_5_pooled": pooled[:5],
    }
    with open(output_dir / "reference.json", "w") as f:
        json.dump(reference_meta, f, indent=2)

    print(f"[✓] Model '{model_id}' successfully converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face models to neural-network-engine format.")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "tinyllamas", "minilm", "tinybert", "whisper", "vit", "modern_bert"], help="Model architecture")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.model}"

    output_dir = Path(args.output_dir)
    if args.model == "gpt2":
        convert_gpt2("gpt2", output_dir)
    elif args.model == "tinyllamas":
        convert_tinyllamas("Xenova/llama2.c-stories15M", output_dir)
    elif args.model == "minilm":
        convert_bert("sentence-transformers/all-MiniLM-L6-v2", output_dir)
    elif args.model == "tinybert":
        convert_tinybert_qa("Intel/dynamic_tinybert", output_dir)
    elif args.model == "whisper":
        convert_whisper("openai/whisper-tiny", output_dir)
    elif args.model == "vit":
        convert_vit("timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k", output_dir)
    elif args.model == "modern_bert":
        convert_modern_bert("answerdotai/ModernBERT-base", output_dir)


if __name__ == "__main__":
    main()
