"""Tests for TorchAlgebraicSimplifyPass."""

import operator
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx

from graph_optimizer.transforms.torch.scalar.algebraic_simplify import (
    TorchAlgebraicSimplifyPass,
)
from ..conftest import count_calls

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_():
    return TorchAlgebraicSimplifyPass()


# ---------------------------------------------------------------------------
# Identity-rule tests — parametrized
# Each entry: (description, forward_fn, target, expected_count_after)
# ---------------------------------------------------------------------------


def _trace(fn):
    class M(torch.nn.Module):
        def forward(self, x):
            return fn(x)

    return fx.symbolic_trace(M())


@pytest.mark.parametrize(
    "fn,target",
    [
        (lambda x: x + 0.0, operator.add),
        (lambda x: 0.0 + x, operator.add),
        (lambda x: x - 0.0, operator.sub),
        (lambda x: x * 1.0, operator.mul),
        (lambda x: 1.0 * x, operator.mul),
        (lambda x: x * 0.0, operator.mul),
        (lambda x: 0.0 * x, operator.mul),
        (lambda x: x / 1.0, operator.truediv),
        (lambda x: x**1, operator.pow),
        (lambda x: x**0, operator.pow),
        (lambda x: torch.neg(torch.neg(x)), torch.neg),
    ],
)
def test_identity_rule_eliminated(pass_, fn, target):
    gm = _trace(fn)
    assert pass_.apply(gm)
    assert count_calls(gm, target) == 0


def test_sub_same_operands(pass_):
    """Sub(x, x) → zeros_like(x)."""
    gm = fx.symbolic_trace(lambda x: x - x)
    assert pass_.apply(gm)
    assert count_calls(gm, operator.sub) == 0


def test_no_change_for_non_identity(pass_):
    gm = _trace(lambda x: x + 2.0)
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Reshape rules
# ---------------------------------------------------------------------------


def test_reshape_chain_fused(pass_):
    """reshape(reshape(x, (2,6)), (3,4)) → single reshape(x, (3,4))."""

    def fn(x):
        return torch.reshape(torch.reshape(x, (2, 6)), (3, 4))

    gm = fx.symbolic_trace(fn)
    assert pass_.apply(gm)
    assert (
        sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.reshape
        )
        == 1
    )


def test_reshape_chain_no_fuse_multiple_users(pass_):
    """Must NOT fuse when intermediate reshape has multiple consumers."""

    def fn(x):
        a = torch.reshape(x, (2, 6))
        b = torch.reshape(a, (3, 4))
        return a, b

    gm = fx.symbolic_trace(fn)
    assert not pass_.apply(gm)


def test_identity_reshape_eliminated_with_meta(pass_):
    """reshape(x, x.shape) is a no-op when shape meta is available."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.reshape(x, (3, 4))

    gm = fx.symbolic_trace(M())
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.zeros(3, 4)
    assert pass_.apply(gm)
    assert (
        sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.reshape
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pass_name(pass_):
    assert pass_.name == "algebraic_simplify"


def test_pass_registry():
    from graph_optimizer.core.passes import PassRegistry

    assert isinstance(
        PassRegistry.get_pass("torch_algebraic_simplify"), TorchAlgebraicSimplifyPass
    )
