"""Shared pytest fixtures for PyTorch transform tests."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx  # noqa: E402

from graph_optimizer.transforms.torch.scalar.algebraic_simplify import (
    TorchAlgebraicSimplifyPass,
)
from graph_optimizer.transforms.torch.scalar.constant_fold import TorchConstantFoldPass
from graph_optimizer.transforms.torch.scalar.cse import TorchCSEPass
from graph_optimizer.transforms.torch.combine.matmul_fuse import MatmulFusePass


@pytest.fixture
def alg_pass():
    return TorchAlgebraicSimplifyPass()


@pytest.fixture
def fold_pass():
    return TorchConstantFoldPass()


@pytest.fixture
def cse_pass():
    return TorchCSEPass()


@pytest.fixture
def matmul_pass():
    return MatmulFusePass()


def count_calls(gm: fx.GraphModule, target) -> int:
    """Count call_function nodes with the given target."""
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and n.target == target
    )
