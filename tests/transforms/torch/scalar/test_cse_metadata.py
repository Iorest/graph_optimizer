"""Tests for TorchCSEPass with device and dtype arguments."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx
from graph_optimizer.transforms.torch.scalar.cse import TorchCSEPass
from ..conftest import count_calls


def test_cse_with_device_and_dtype():
    pass_ = TorchCSEPass()

    def fn(x):
        # Create same tensor with explicit device/dtype
        # These should be deduped now
        t1 = torch.add(x, 1.0, alpha=1)
        t2 = torch.add(x, 1.0, alpha=1)
        # Use device/dtype/layout/memory_format
        t3 = torch.empty_like(
            x, dtype=torch.float32, device="cpu", layout=torch.strided
        )
        t4 = torch.empty_like(
            x, dtype=torch.float32, device="cpu", layout=torch.strided
        )
        return t1 + t2 + t3 + t4

    gm = fx.symbolic_trace(fn)
    # Before optimization
    assert count_calls(gm, torch.add) == 2
    assert count_calls(gm, torch.empty_like) == 2

    # Apply pass
    assert pass_.apply(gm)

    # After optimization
    assert count_calls(gm, torch.add) == 1
    assert count_calls(gm, torch.empty_like) == 1


def test_cse_avoids_different_metadata():
    pass_ = TorchCSEPass()

    def fn(x):
        # Different dtypes -> should NOT be deduped
        t1 = torch.empty_like(x, dtype=torch.float32)
        t2 = torch.empty_like(x, dtype=torch.float64)
        return t1 + t2

    gm = fx.symbolic_trace(fn)
    assert not pass_.apply(gm)
    assert count_calls(gm, torch.empty_like) == 2
