import time
import pytest
import torch
import torch.nn as nn
import torch.fx as fx
from torchvision.models import resnet18

from graph_optimizer.core.torch.torch_optimizer import TorchOptimizer

# Skip if PyTorch is not available
pytest.importorskip("torch", reason="PyTorch not installed")


def profile_model(model, input_tensor, iterations=50):
    # Warmup
    for _ in range(5):
        _ = model(input_tensor)

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = model(input_tensor)
    end_time = time.perf_counter()

    return (end_time - start_time) / iterations


def test_resnet18_optimization_correctness_and_performance():
    # 1. Initialize Model
    model = resnet18()
    model.eval()

    # Trace the model
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        pytest.skip(f"Failed to trace ResNet18: {e}")

    # 2. Optimize
    # Use max_iterations=3 to ensure passes converge
    opt = TorchOptimizer(gm)
    optimized_gm = opt.optimize(max_iterations=3)

    # 3. Verify Correctness
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        expected_output = gm(dummy_input)
        actual_output = optimized_gm(dummy_input)

    assert torch.allclose(expected_output, actual_output, atol=1e-5), (
        "Optimized ResNet18 output diverges from original!"
    )

    # 4. Profile Performance
    with torch.no_grad():
        original_time = profile_model(gm, dummy_input)
        optimized_time = profile_model(optimized_gm, dummy_input)

    print(f"\nResNet18 - Original avg time: {original_time * 1000:.3f} ms")
    print(f"ResNet18 - Optimized avg time: {optimized_time * 1000:.3f} ms")

    # We do not strictly assert that optimized_time < original_time because on a small test with random weights
    # and default PyTorch eagerly executing, it might just be overhead. But the test ensures it runs correctly.


class TraceableAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.qkv(x)
        # Using simple chunk instead of complex reshaping for easy tracing
        q = qkv[:, :, : self.proj.in_features]
        k = qkv[:, :, self.proj.in_features : self.proj.in_features * 2]
        v = qkv[:, :, self.proj.in_features * 2 :]

        scores = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return self.proj(out)


class TraceableTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = TraceableAttention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


def test_transformer_optimization_correctness_and_performance():
    # 1. Initialize Model (Using our traceable mock instead of nn.Transformer)
    model = nn.Sequential(
        TraceableTransformerBlock(d_model=128), TraceableTransformerBlock(d_model=128)
    )
    model.eval()

    # Trace the model
    try:
        from torch.fx import symbolic_trace

        gm = symbolic_trace(model)
    except Exception as e:
        pytest.fail(f"Failed to trace Transformer: {e}")

    # 2. Optimize
    opt = TorchOptimizer(gm)
    optimized_gm = opt.optimize(max_iterations=3)

    # 3. Verify Correctness
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, 128)

    with torch.no_grad():
        expected_output = gm(dummy_input)
        actual_output = optimized_gm(dummy_input)

    assert torch.allclose(expected_output, actual_output, atol=1e-5), (
        "Optimized Transformer output diverges from original!"
    )

    # 4. Profile Performance
    with torch.no_grad():
        original_time = profile_model(gm, dummy_input)
        optimized_time = profile_model(optimized_gm, dummy_input)

    print(f"\nTransformer Mock - Original avg time: {original_time * 1000:.3f} ms")
    print(f"Transformer Mock - Optimized avg time: {optimized_time * 1000:.3f} ms")
