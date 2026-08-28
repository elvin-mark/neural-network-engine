"""
Python Example 03: Transformer Language Model (nanoGPT style) with Causal Self-Attention.

Demonstrates:
1. Token & Position Embeddings, Multi-Head Causal Attention, and LayerNorm.
2. Autoregressive language modeling with CrossEntropyLoss and Adam optimization.
"""

import sys
import time
from pathlib import Path

# Add python module directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_network_engine as nne
import numpy as np


def main():
    print("=" * 65)
    print(" Python Example 03: Transformer Language Model with Rust Backend")
    print("=" * 65 + "\n")

    # 1. Hyperparameters
    vocab_size = 128
    max_seq_len = 32
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 4
    seq_len = 16

    print("Transformer Architecture:")
    print(f"  • Vocab Size:   {vocab_size} tokens")
    print(f"  • Max Seq Len:  {max_seq_len}")
    print(f"  • Model Dim:    {d_model}")
    print(f"  • Heads:        {num_heads} (head_dim = {d_model // num_heads})")
    print(f"  • Layers:       {num_layers} decoder blocks\n")

    # 2. Instantiate TransformerLM
    model = nne.TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    params = model.parameters()
    print(f"Model initialized with {len(params)} trainable parameter tensors.")

    # 3. Create synthetic sequence dataset
    np.random.seed(42)
    input_tokens_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.float32)
    target_tokens_np = np.random.randint(0, vocab_size, size=(batch_size * seq_len,)).astype(np.float32)

    input_tokens = nne.Tensor.from_numpy(input_tokens_np, requires_grad=False)
    target_tokens = nne.Tensor.from_numpy(target_tokens_np, requires_grad=False)

    # 4. Optimizer & Loss
    optimizer = nne.Adam(params, lr=1e-3)
    loss_fn = nne.CrossEntropyLoss()

    print("Training TransformerLM for 20 steps...\n")

    start_time = time.time()
    for step in range(1, 21):
        optimizer.zero_grad()

        # Forward pass: shape [B, T, VocabSize]
        logits = model(input_tokens)

        # Flatten to [B*T, VocabSize] for CrossEntropyLoss
        logits_flat = logits.reshape([batch_size * seq_len, vocab_size])
        loss = loss_fn(logits_flat, target_tokens)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:2d}/20 | CrossEntropyLoss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\nTransformer training completed in {elapsed * 1000:.2f} ms ({elapsed * 1000 / 20:.2f} ms/step)!")
    print("Python Transformer LM example executed successfully!\n")


if __name__ == "__main__":
    main()
