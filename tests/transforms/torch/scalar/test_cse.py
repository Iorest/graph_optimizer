"""Tests for TorchCSEPass (Common Subexpression Elimination)."""

import operator
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx

from graph_optimizer.transforms.torch.scalar.cse import TorchCSEPass
from ..conftest import count_calls

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_():
    return TorchCSEPass()


# ---------------------------------------------------------------------------
# Basic deduplication
# ---------------------------------------------------------------------------


def test_eliminates_duplicate_add(pass_):
    def fn(x, y):
        return operator.add(x, y), operator.add(x, y)

    gm = fx.symbolic_trace(fn)
    assert count_calls(gm, operator.add) == 2
    assert pass_.apply(gm)
    assert count_calls(gm, operator.add) == 1


def test_eliminates_duplicate_mul(pass_):
    def fn(x, y):
        return operator.mul(x, y) + operator.mul(x, y)

    gm = fx.symbolic_trace(fn)
    assert pass_.apply(gm)
    assert count_calls(gm, operator.mul) == 1


# ---------------------------------------------------------------------------
# Chain deduplication
# ---------------------------------------------------------------------------


def test_chain_deduplication(pass_):
    def fn(x):
        t1 = operator.neg(x)
        t2 = operator.neg(x)
        return operator.add(t1, t1), operator.add(t2, t2)

    gm = fx.symbolic_trace(fn)
    pass_.apply(gm)
    pass_.apply(gm)  # second pass catches downstream dups
    assert count_calls(gm, operator.neg) == 1


# ---------------------------------------------------------------------------
# Non-merging cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,changed",
    [
        (
            lambda x, y, z: (operator.add(x, y), operator.add(x, z)),
            False,
        ),  # different args
        (
            lambda x, y: (operator.add(x, y), operator.mul(x, y)),
            False,
        ),  # different targets
        (lambda x, y: operator.add(x, y), False),  # no duplicates
    ],
)
def test_no_merge_cases(pass_, fn, changed):
    gm = fx.symbolic_trace(fn)
    assert pass_.apply(gm) == changed


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent(pass_):
    def fn(x, y):
        return operator.add(x, y), operator.add(x, y)

    gm = fx.symbolic_trace(fn)
    pass_.apply(gm)
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pass_name(pass_):
    assert pass_.name == "cse"


def test_pass_registry():
    from graph_optimizer.core.passes import PassRegistry

    assert isinstance(PassRegistry.get_pass("torch_cse"), TorchCSEPass)


def test_priority_higher_than_algebraic():
    from graph_optimizer.core.passes import PassRegistry

    meta = PassRegistry._pass_metadata
    assert meta["torch_cse"]["priority"] > meta["torch_algebraic_simplify"]["priority"]
