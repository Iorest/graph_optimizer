"""
PyTorch FX Pipeline Demo
=========================
Mirrors demos/run_demo.py (TensorFlow version).

What this demo does
-------------------
1. Builds a synthetic ``torch.nn.Module`` that deliberately contains:
   - Redundant subexpressions (caught by CSE)
   - Algebraic identities: Add(x, 0), Mul(x, 1), Sub(x, x) (simplified)
   - A double-reshape chain (chain-fused)
   - Constant-only sub-graph (folded by constant_fold)
2. Traces it with ``torch.fx.symbolic_trace``
3. Runs ``OptimizationPipeline`` (Torch backend) at opt_level=1
4. Prints the ``OptimizationReport`` summary table
5. Verifies numerical consistency across N random inputs
"""

import os
import sys
import site

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# We add the *parent* of the graph_optimizer package directory (i.e. Workspace),
# NOT the repo root itself. This lets `import graph_optimizer` work while
# NOT making `transforms/torch/` importable as bare `torch` (which would
# shadow the installed PyTorch package).
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_workspace_parent = os.path.dirname(_workspace)
if _workspace_parent not in sys.path:
    sys.path.insert(0, _workspace_parent)

# Also ensure the user-installed site-packages (pip install --user) are visible
for _p in (
    site.getusersitepackages()
    if isinstance(site.getusersitepackages(), list)
    else [site.getusersitepackages()]
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.fx as fx

from graph_optimizer.runner import OptimizationPipeline
from graph_optimizer.utils.logger import set_log_level, INFO

set_log_level(INFO)

# ---------------------------------------------------------------------------
# Demo model
# ---------------------------------------------------------------------------


class DemoModel(nn.Module):
    """
    A deliberately suboptimal model with patterns every Torch pass targets:

    Pass                 | Pattern in forward()
    -------------------- | ----------------------------------
    constant_fold        | self.scale * self.offset  (compile-time constants)
    algebraic_simplify   | x + 0, x * 1, x - x, reshape chain
    cse                  | (x + y) computed twice
    """

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Constant-fold candidates: scale and offset are scalar Python attrs
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("bias_zero", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Linear projection
        h = self.linear(x)  # (B, out_features)

        # ── CSE target: compute same expression twice
        a = torch.relu(h)
        b = torch.relu(h)  # duplicate of a

        # ── Algebraic identity targets
        c = a + self.bias_zero  # Add(x, 0) → x
        d = b * 1.0  # Mul(x, 1) → x (via scalar)
        e = c - c  # Sub(x, x) → 0

        # ── Reshape chain: (B, out) → (B*out,) → (B, out) [identity]
        flat = torch.reshape(a, (-1,))  # dynamic shape — chain only
        back = torch.reshape(flat, (x.shape[0], -1))

        # ── Combine everything into a single output
        out = d + back + e
        return out


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------


def run_consistency_tests(
    original_fn,
    optimized_gm,
    num_tests: int = 20,
    batch_size: int = 2,
    in_features: int = 8,
):
    """
    Run N random-input tests comparing original model vs optimised graph.
    Mirrors the TF demo's run_consistency_tests().
    """
    print(f"\n{'=' * 60}")
    print(f"Running {num_tests} consistency tests with random inputs...")
    print(f"{'=' * 60}")

    all_passed = True
    max_diffs = []

    for i in range(num_tests):
        torch.manual_seed(42 + i * 1000)
        x = torch.randn(batch_size, in_features)

        with torch.no_grad():
            expected = original_fn(x)
            actual = optimized_gm(x)

        diff = (expected - actual).abs().max().item()
        passed = diff < 1e-5
        max_diffs.append(diff)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"\n  Test {i + 1:2d}: {status}  (seed={42 + i * 1000})")
        print(f"    Output shape:  {expected.shape}")
        print(f"    Expected[:5]:  {expected.flatten()[:5].tolist()}")
        print(f"    Optimized[:5]: {actual.flatten()[:5].tolist()}")
        print(f"    Max diff:      {diff:.2e}")

    print(f"\n{'=' * 60}")
    print(f"Summary: {sum(1 for d in max_diffs if d < 1e-5)}/{num_tests} tests passed")
    print(f"Max diff across all tests:  {max(max_diffs):.2e}")
    print(f"Mean diff across all tests: {sum(max_diffs) / len(max_diffs):.2e}")
    print(f"{'=' * 60}")
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 60)
    print("  PyTorch FX Optimization Pipeline — Demo")
    print("=" * 60)

    IN_FEATURES = 8
    OUT_FEATURES = 4
    BATCH = 2

    # Build model
    torch.manual_seed(0)
    model = DemoModel(in_features=IN_FEATURES, out_features=OUT_FEATURES)
    model.eval()

    # Trace the FX graph
    gm = fx.symbolic_trace(model)
    original_node_count = len(gm.graph.nodes)

    print(f"\nOriginal FX graph: {original_node_count} nodes")
    print("\nOriginal graph IR:")
    gm.print_readable()

    # Keep original (unoptimised) callable for consistency tests
    original_model = model  # nn.Module — uses eager, no FX

    # Run OptimizationPipeline (Torch backend)
    print("\nStarting Torch OptimizationPipeline...")
    pipeline = OptimizationPipeline(
        graph_module=gm,
        level=1,  # opt_level=1: constant_fold, algebraic_simplify, cse
        debug=False,
    )
    report = pipeline.run()
    report.print_summary()

    # Optimised graph stats
    optimized_node_count = len(gm.graph.nodes)
    removed = original_node_count - optimized_node_count
    pct = 100 * removed / original_node_count if original_node_count else 0

    print(f"\nOptimized FX graph: {optimized_node_count} nodes")
    print(f"Reduction: {removed} nodes removed ({pct:.1f}%)")
    print("\nOptimized graph IR:")
    gm.print_readable()

    # Consistency tests
    all_passed = run_consistency_tests(
        original_model,
        gm,
        num_tests=20,
        batch_size=BATCH,
        in_features=IN_FEATURES,
    )

    print()
    if all_passed:
        print("✓ All consistency tests passed!")
        return 0
    else:
        print("✗ Some consistency tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
