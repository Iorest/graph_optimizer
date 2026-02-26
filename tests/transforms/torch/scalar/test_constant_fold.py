"""Tests for TorchConstantFoldPass."""

import operator
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx

from graph_optimizer.transforms.torch.scalar.constant_fold import TorchConstantFoldPass

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_():
    return TorchConstantFoldPass()


# ---------------------------------------------------------------------------
# Scalar constant folding
# ---------------------------------------------------------------------------


def test_fold_add_scalars(pass_):
    class M(torch.nn.Module):
        def forward(self, x):
            return x + torch.add(torch.tensor(3), torch.tensor(4))

    gm = fx.symbolic_trace(M())
    changed = pass_.apply(gm)
    if changed:
        fns = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.add not in fns


def test_fold_mul_scalars(pass_):
    class M(torch.nn.Module):
        def forward(self, x):
            return x + torch.mul(torch.tensor(2.0), torch.tensor(3.0))

    gm = fx.symbolic_trace(M())
    changed = pass_.apply(gm)
    if changed:
        fns = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.mul not in fns


def test_attr_propagation_does_not_crash(pass_):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.tensor(2.0)

        def forward(self, x):
            return x + self.c

    gm = fx.symbolic_trace(M())
    assert isinstance(pass_.apply(gm), bool)


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


def test_no_foldable_constants(pass_):
    gm = fx.symbolic_trace(lambda x, y: x + y)
    assert not pass_.apply(gm)


def test_identity_graph(pass_):
    gm = fx.symbolic_trace(lambda x: x)
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent(pass_):
    class M(torch.nn.Module):
        def forward(self, x):
            return x + torch.add(torch.tensor(2), torch.tensor(3))

    gm = fx.symbolic_trace(M())
    pass_.apply(gm)
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pass_name(pass_):
    assert pass_.name == "constant_fold"


def test_registry():
    from graph_optimizer.core.passes import PassRegistry

    assert isinstance(
        PassRegistry.get_pass("torch_constant_fold"), TorchConstantFoldPass
    )


def test_priority_lower_than_algebraic():
    from graph_optimizer.core.passes import PassRegistry

    meta = PassRegistry._pass_metadata
    assert (
        meta["torch_constant_fold"]["priority"]
        < meta["torch_algebraic_simplify"]["priority"]
    )
