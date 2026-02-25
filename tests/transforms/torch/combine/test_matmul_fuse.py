"""Tests for MatmulFusePass."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.fx as fx

from graph_optimizer.transforms.torch.combine.matmul_fuse import MatmulFusePass

# ---------------------------------------------------------------------------
# Fixture / helper
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_():
    return MatmulFusePass()


def _chain(d_in, d_mid, d_out):
    """Traced module: y = matmul(matmul(x, A), B) with constant A, B."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.randn(d_in, d_mid)
            self.B = torch.randn(d_mid, d_out)

        def forward(self, x):
            return torch.matmul(torch.matmul(x, self.A), self.B)

    return fx.symbolic_trace(M())


def _count_matmuls(gm):
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.matmul
    )


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------


def test_fuses_constant_weight_chain(pass_):
    gm = _chain(4, 8, 6)
    assert _count_matmuls(gm) == 2
    assert pass_.apply(gm)
    assert _count_matmuls(gm) == 1


def test_result_numerically_correct(pass_):
    gm = _chain(4, 8, 6)
    x = torch.randn(2, 4)
    expected = x @ getattr(gm, "A") @ getattr(gm, "B")
    pass_.apply(gm)
    assert torch.allclose(gm(x), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Non-fusable guards
# ---------------------------------------------------------------------------


def test_no_fuse_dynamic_right_operand(pass_):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.randn(4, 8)

        def forward(self, x, B):
            return torch.matmul(torch.matmul(x, self.A), B)

    gm = fx.symbolic_trace(M())
    assert not pass_.apply(gm)
    assert _count_matmuls(gm) == 2


def test_no_fuse_intermediate_has_multiple_users(pass_):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.randn(4, 8)
            self.B = torch.randn(8, 6)

        def forward(self, x):
            mid = torch.matmul(x, self.A)
            return mid, torch.matmul(mid, self.B)

    gm = fx.symbolic_trace(M())
    assert not pass_.apply(gm)


def test_no_fuse_shape_mismatch(pass_):
    """A=(4,8) @ B=(6,4) is invalid — should NOT fuse."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.randn(4, 8)
            self.B = torch.randn(6, 4)

        def forward(self, x):
            return torch.matmul(torch.matmul(x, self.A), self.B)

    gm = fx.symbolic_trace(M())
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent(pass_):
    gm = _chain(4, 8, 6)
    pass_.apply(gm)
    assert not pass_.apply(gm)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pass_name(pass_):
    assert pass_.name == "matmul_fuse"


def test_pass_registry():
    from graph_optimizer.core.passes import PassRegistry

    assert isinstance(PassRegistry.get_pass("torch_matmul_fuse"), MatmulFusePass)


def test_opt_level_is_2():
    from graph_optimizer.core.passes import PassRegistry

    assert PassRegistry._pass_metadata["torch_matmul_fuse"]["opt_level"] == 2


def test_priority_ordering():
    from graph_optimizer.core.passes import PassRegistry

    meta = PassRegistry._pass_metadata
    assert (
        meta["torch_matmul_fuse"]["priority"]
        > meta["torch_algebraic_simplify"]["priority"]
    )
    assert meta["torch_matmul_fuse"]["priority"] > meta["torch_cse"]["priority"]
