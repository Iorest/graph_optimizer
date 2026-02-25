import json
from typing import Dict, Any


class OptimizationReport:
    """
    Structured report for optimization results.
    """

    def __init__(
        self,
        initial_nodes: int,
        final_nodes: int,
        total_time: float,
        pass_stats: Dict[str, Any],
        graph_def: Any = None,  # TF backend — GraphDef
        graph_module: Any = None,  # Torch backend — fx.GraphModule
    ):
        self.initial_nodes = initial_nodes
        self.final_nodes = final_nodes
        self.total_time = total_time
        self.pass_stats = pass_stats
        # Unified graph reference (whichever backend set it)
        self.graph = graph_module if graph_module is not None else graph_def
        self.graph_def = self.graph  # TF-style alias
        self.graph_module = self.graph  # Torch-style alias
        self.nodes_removed = initial_nodes - final_nodes
        self.reduction_pct = (
            (self.nodes_removed / initial_nodes * 100) if initial_nodes > 0 else 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON export."""
        return {
            "summary": {
                "initial_nodes": self.initial_nodes,
                "final_nodes": self.final_nodes,
                "nodes_removed": self.nodes_removed,
                "reduction_pct": round(self.reduction_pct, 2),
                "total_time_sec": round(self.total_time, 4),
            },
            "passes": self.pass_stats,
        }

    def save_json(self, path: str):
        """Save report as JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Prints a human-readable summary (Markdown style)."""
        print("\n" + "=" * 70)
        print("OPTIMIZATION REPORT")
        print("=" * 70)

        print("\nOverall:")
        print(
            f"  Nodes: {self.initial_nodes} -> {self.final_nodes} (Reduction: {self.reduction_pct:.1f}%)"
        )
        print(f"  Total Time: {self.total_time:.3f}s")
        print(f"  Total Passes: {len(self.pass_stats)}")

        print("\nPer-Pass Breakdown:")
        print("-" * 70)
        print(f"{'Pass':<30} {'Iters':>6} {'Changes':>8} {'Nodes':>15} {'Time':>8}")
        print("-" * 70)

        for name, stats in self.pass_stats.items():
            iters = len(stats.get("iterations", []))
            changes = stats.get("total_changes", 0)
            nodes = f"{stats.get('nodes_before', 0)}->{stats.get('nodes_after', 0)}"
            time = f"{stats.get('duration', 0):.3f}s"
            print(f"  {name:<28} {iters:>6} {changes:>8} {nodes:>15} {time:>8}")

        print("-" * 70)
        print("=" * 70 + "\n")
