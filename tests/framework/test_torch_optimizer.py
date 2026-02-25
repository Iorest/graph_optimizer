"""
Tests for TorchOptimizer — the PyTorch FX optimization pipeline.

Covers: pass lifecycle, convergence loop, rollback on exception,
default pass name resolution, and debug dump.
"""

import pytest
import torch
import torch.fx as fx
from graph_optimizer.core.base_pass import BaseOptimizationPass
from graph_optimizer.core.torch.torch_optimizer import (
    TorchOptimizer,
    _snapshot,
    _restore,
    _default_pass_names,
)

pytest.importorskip("torch", reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_gm() -> fx.GraphModule:
    """Return a trivial GraphModule: f(x) = x + 0."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x + 0.0

    return fx.symbolic_trace(M())


def _identity_gm() -> fx.GraphModule:
    """Return a GraphModule: f(x) = x (already optimal, no rules to fire)."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x

    return fx.symbolic_trace(M())


# ---------------------------------------------------------------------------
# _snapshot / _restore
# ---------------------------------------------------------------------------


def test_snapshot_is_independent_copy():
    gm = _simple_gm()
    snap = _snapshot(gm)
    # Modify the live graph
    with gm.graph.inserting_before(list(gm.graph.nodes)[0]):
        gm.graph.placeholder("injected")
    gm.recompile()
    # Restore
    _restore(gm, snap)
    names = [n.name for n in gm.graph.nodes]
    assert "injected" not in names


def test_restore_makes_graph_compilable():
    gm = _simple_gm()
    snap = _snapshot(gm)
    _restore(gm, snap)
    # Should not raise
    x = torch.ones(2, 2)
    gm(x)


# ---------------------------------------------------------------------------
# _default_pass_names
# ---------------------------------------------------------------------------


def test_default_pass_names_all_torch_prefixed():
    names = _default_pass_names(opt_level=1)
    assert all(n.startswith("torch_") for n in names), names


def test_default_pass_names_nonempty():
    names = _default_pass_names(opt_level=1)
    assert len(names) > 0


# ---------------------------------------------------------------------------
# TorchOptimizer construction
# ---------------------------------------------------------------------------


def test_default_construction_uses_registered_passes():
    gm = _identity_gm()
    opt = TorchOptimizer(gm, opt_level=1)
    assert len(opt.passes) > 0


def test_explicit_passes_subset():
    gm = _identity_gm()
    opt = TorchOptimizer(gm, passes=["torch_algebraic_simplify"])
    assert len(opt.passes) == 1
    assert "algebraic_simplify" in opt.passes[0].name.lower()


def test_node_count_property():
    gm = _simple_gm()
    opt = TorchOptimizer(gm)
    assert opt.node_count == len(list(gm.graph.nodes))


# ---------------------------------------------------------------------------
# optimize() end-to-end
# ---------------------------------------------------------------------------


def test_optimize_simplifies_add_zero():
    """add(x, 0.0) should be eliminated by AlgebraicSimplify."""
    gm = _simple_gm()
    opt = TorchOptimizer(gm, passes=["torch_algebraic_simplify"])
    result = opt.optimize()
    calls = [n for n in result.graph.nodes if n.op == "call_function"]
    assert len(calls) == 0, f"Expected no call_function nodes, got {calls}"


def test_optimize_returns_graph_module():
    gm = _identity_gm()
    opt = TorchOptimizer(gm)
    result = opt.optimize()
    assert isinstance(result, fx.GraphModule)


def test_optimize_converges_on_optimal_graph():
    """Optimizer on an already-optimal graph should converge in 1 iteration."""
    gm = _identity_gm()
    opt = TorchOptimizer(gm)
    result = opt.optimize(max_iterations=10)
    assert result is gm  # same object returned


def test_optimize_graph_is_executable_after():
    gm = _simple_gm()
    opt = TorchOptimizer(gm, passes=["torch_algebraic_simplify"])
    result = opt.optimize()
    x = torch.tensor([1.0, 2.0, 3.0])
    out = result(x)
    assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Rollback on pass exception
# ---------------------------------------------------------------------------


def test_rollback_on_pass_exception():
    """A pass that crashes must not corrupt the graph."""

    class BoomPass(BaseOptimizationPass):
        @property
        def name(self):
            return "boom"

        def apply(self, gm):
            with gm.graph.inserting_before(list(gm.graph.nodes)[0]):
                gm.graph.placeholder("BAD_NODE")
            raise RuntimeError("intentional boom")

    gm = _simple_gm()
    opt = TorchOptimizer(gm, passes=["torch_algebraic_simplify"])
    opt.passes = [BoomPass(), opt.passes[0]]
    result = opt.optimize()
    node_names = [n.name for n in result.graph.nodes]
    assert "BAD_NODE" not in node_names


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------


def test_debug_dump_creates_files(tmp_path):
    gm = _simple_gm()
    opt = TorchOptimizer(gm, passes=["torch_algebraic_simplify"])
    opt.optimize(debug_dir=str(tmp_path))
    files = list(tmp_path.iterdir())
    assert len(files) > 0
    assert all(f.suffix == ".txt" for f in files)
