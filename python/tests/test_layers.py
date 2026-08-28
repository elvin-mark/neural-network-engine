"""
Unit and Integration Tests for Python Bindings (`neural_network_engine`).
"""

import sys
from pathlib import Path

# Add python module directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_network_engine as nne
import numpy as np


def test_tensor_numpy_roundtrip():
    arr = np.random.randn(4, 8, 16).astype(np.float32)
    t = nne.Tensor.from_numpy(arr, requires_grad=True)
    assert t.shape == [4, 8, 16]
    assert t.requires_grad is True

    back = t.to_numpy()
    assert np.allclose(arr, back)
    print("✓ test_tensor_numpy_roundtrip passed")


def test_tensor_arithmetic_and_autograd():
    a_np = np.array([[2.0, 3.0]], dtype=np.float32)
    b_np = np.array([[4.0, 5.0]], dtype=np.float32)

    a = nne.Tensor.from_numpy(a_np, requires_grad=True)
    b = nne.Tensor.from_numpy(b_np, requires_grad=True)

    c = (a * b).sum()
    c.backward()

    # dc/da = b, dc/db = a
    assert np.allclose(a.grad.to_numpy(), b_np)
    assert np.allclose(b.grad.to_numpy(), a_np)
    print("✓ test_tensor_arithmetic_and_autograd passed")


def test_conv2d_and_maxpool2d():
    x = nne.Tensor.randn([2, 3, 32, 32], 0.0, 1.0, requires_grad=True)
    conv = nne.Conv2d(3, 16, (3, 3), padding=(1, 1))
    pool = nne.MaxPool2d((2, 2))

    out = pool(conv(x))
    assert out.shape == [2, 16, 16, 16]

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == [2, 3, 32, 32]
    print("✓ test_conv2d_and_maxpool2d passed")


def test_batchnorm_1d_and_2d():
    # BatchNorm1d
    bn1 = nne.BatchNorm1d(64)
    x1 = nne.Tensor.randn([8, 64], 0.0, 1.0, requires_grad=True)
    out1 = bn1(x1)
    assert out1.shape == [8, 64]

    # BatchNorm2d
    bn2 = nne.BatchNorm2d(16)
    x2 = nne.Tensor.randn([4, 16, 8, 8], 0.0, 1.0, requires_grad=True)
    out2 = bn2(x2)
    assert out2.shape == [4, 16, 8, 8]
    print("✓ test_batchnorm_1d_and_2d passed")


def test_embedding():
    emb = nne.Embedding(100, 32)
    tokens = nne.Tensor.from_numpy(np.array([[0, 5, 99], [12, 45, 67]], dtype=np.float32), requires_grad=False)
    out = emb(tokens)
    assert out.shape == [2, 3, 32]
    print("✓ test_embedding passed")


def test_attention_and_flash_attention():
    x = nne.Tensor.randn([2, 8, 32], 0.0, 1.0, requires_grad=True)
    
    # Standard MultiHeadAttention
    mha = nne.MultiHeadAttention(32, 4, is_causal=True)
    out_mha = mha(x)
    assert out_mha.shape == [2, 8, 32]

    # FlashAttention-2
    fa = nne.FlashAttention(32, 4, is_causal=True)
    out_fa = fa(x)
    assert out_fa.shape == [2, 8, 32]
    print("✓ test_attention_and_flash_attention passed")


def test_swiglu_and_moe_layer():
    x = nne.Tensor.randn([4, 16, 32], 0.0, 1.0, requires_grad=True)

    # SwiGLU
    swiglu = nne.SwiGLU(32, 64)
    out_sw = swiglu(x)
    assert out_sw.shape == [4, 16, 32]

    # MoE Layer
    moe = nne.MoELayer(32, 64, num_experts=4, top_k=2, aux_loss_coeff=0.01)
    out_moe, aux_loss = moe.forward_with_aux(x)
    assert out_moe.shape == [4, 16, 32]
    assert aux_loss.shape == []
    assert aux_loss.item() >= 0.0
    print("✓ test_swiglu_and_moe_layer passed")


def test_transformer_block_and_lm():
    seq = nne.Tensor.randn([2, 8, 32], 0.0, 1.0, requires_grad=True)
    block = nne.TransformerBlock(32, 4, is_causal=True)
    out = block(seq)
    assert out.shape == [2, 8, 32]

    lm = nne.TransformerLM(vocab_size=50, max_seq_len=16, d_model=32, num_heads=4, num_layers=2)
    toks = nne.Tensor.from_numpy(np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32), requires_grad=False)
    logits = lm(toks)
    assert logits.shape == [2, 4, 50]
    print("✓ test_transformer_block_and_lm passed")


def test_resnet18_and_residual_block():
    res_block = nne.ResidualBlock(16, 32, stride=2)
    x = nne.Tensor.randn([2, 16, 16, 16], 0.0, 1.0, requires_grad=True)
    out_block = res_block(x)
    assert out_block.shape == [2, 32, 8, 8]

    resnet = nne.ResNet18(num_classes=10, in_channels=3, cifar_stem=True)
    x_img = nne.Tensor.randn([2, 3, 32, 32], 0.0, 1.0, requires_grad=True)
    logits = resnet(x_img)
    assert logits.shape == [2, 10]
    print("✓ test_resnet18_and_residual_block passed")


def test_losses_and_optimizers():
    linear = nne.Linear(16, 4)
    x = nne.Tensor.randn([8, 16], 0.0, 1.0, requires_grad=False)
    logits = linear(x)

    # CrossEntropyLoss
    ce = nne.CrossEntropyLoss()
    targets = nne.Tensor.from_numpy(np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32), requires_grad=False)
    loss = ce(logits, targets)
    assert loss.item() > 0.0

    # Dynamic LossScaler with Adam
    scaler = nne.LossScaler(1024.0)
    opt = nne.Adam(linear.parameters(), lr=0.01)

    opt.zero_grad()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    stepped = scaler.step_adam(opt)
    assert stepped is True
    print("✓ test_losses_and_optimizers passed")


if __name__ == "__main__":
    print("========================================")
    print(" Running Python Bindings Test Suite")
    print("========================================")
    test_tensor_numpy_roundtrip()
    test_tensor_arithmetic_and_autograd()
    test_conv2d_and_maxpool2d()
    test_batchnorm_1d_and_2d()
    test_embedding()
    test_attention_and_flash_attention()
    test_swiglu_and_moe_layer()
    test_transformer_block_and_lm()
    test_resnet18_and_residual_block()
    test_losses_and_optimizers()
    print("========================================")
    print(" All 10 Python Test Suites Passed (100% OK)!")
    print("========================================")
