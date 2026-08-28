"""
Python Example 04: Mixture of Experts (MoE) & FlashAttention-2 Acceleration.

Demonstrates:
1. Fast FlashAttention-2 vs Standard Attention in Python.
2. Sparse Top-2 Mixture of Experts (MoE) routing with SwiGLU experts and auxiliary load-balancing loss.
3. Training with Dynamic Loss Scaling (AMP LossScaler).
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
    print(" Python Example 04: Mixture of Experts & FlashAttention-2 Demo")
    print("=" * 65 + "\n")

    batch_size = 4
    seq_len = 32
    d_model = 64
    num_heads = 4
    num_experts = 8
    top_k = 2

    # 1. FlashAttention-2 vs MultiHeadAttention Comparison
    print("-----------------------------------------------------------------")
    print(" 1. Attention Benchmark: FlashAttention-2 vs MultiHeadAttention")
    print("-----------------------------------------------------------------")

    seq = nne.Tensor.randn([batch_size, seq_len, d_model], 0.0, 1.0, requires_grad=True)

    mha = nne.MultiHeadAttention(d_model, num_heads, is_causal=True)
    fa = nne.FlashAttention(d_model, num_heads, is_causal=True)

    t0 = time.time()
    for _ in range(50):
        _ = mha(seq)
    mha_time = (time.time() - t0) * 1000 / 50

    t0 = time.time()
    for _ in range(50):
        _ = fa(seq)
    fa_time = (time.time() - t0) * 1000 / 50

    print(f"  • Standard MultiHeadAttention: {mha_time:.3f} ms / forward pass")
    print(f"  • FlashAttention-2 (Tiled):     {fa_time:.3f} ms / forward pass")
    print(f"  • Speedup:                     {mha_time / fa_time:.2f}x faster with O(T) memory\n")

    # 2. Sparse Mixture of Experts (MoE) Routing & Training
    print("-----------------------------------------------------------------")
    print(" 2. Sparse Mixture of Experts (MoE) Training Loop")
    print("-----------------------------------------------------------------")
    print(f"  • MoE Configuration: {num_experts} SwiGLU Experts, Top-{top_k} Routing per token")

    moe = nne.MoELayer(
        d_model=d_model,
        hidden_dim=d_model * 2,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_coeff=0.01,
    )
    params = moe.parameters()

    optimizer = nne.Adam(params, lr=1e-3)
    scaler = nne.LossScaler(1024.0)

    target_seq = nne.Tensor.randn([batch_size, seq_len, d_model], 0.0, 1.0, requires_grad=False)

    print("Training MoE layer for 20 steps with Dynamic LossScaler (AMP)...\n")

    for step in range(1, 21):
        optimizer.zero_grad()

        moe_out, aux_loss = moe.forward_with_aux(seq)
        diff = moe_out - target_seq
        task_loss = (diff * diff).mean()
        total_loss = task_loss + aux_loss

        scaled_loss = scaler.scale(total_loss)
        scaled_loss.backward()
        scaler.step_adam(optimizer)

        if step % 5 == 0 or step == 1:
            print(
                f"  Step {step:2d}/20 | Task Loss: {task_loss.item():.4f} | "
                f"Aux Loss: {aux_loss.item():.5f} | Scale: {scaler.current_scale():.1f}"
            )

    print("\nPython MoE and FlashAttention example completed successfully!\n")


if __name__ == "__main__":
    main()
