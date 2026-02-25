from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    """
    Abstract base class for framework-specific graph optimizers.
    All backend-specific optimizers (e.g., TFGraphOptimizer, TorchOptimizer)
    must inherit from this and implement the optimize() method.
    """

    def __init__(self, graph: Any, passes: list[str] = None, opt_level: int = 1):
        """
        Initialize the optimizer.

        Args:
            graph: The backend-specific graph representation.
            passes: Optional list of pass names to execute. If None, defaults determined by opt_level.
            opt_level: Optimization level (e.g. 1 or 2).
        """
        self.graph = graph
        self.passes = passes
        self.opt_level = opt_level

    @abstractmethod
    def optimize(self, context=None, debug_dir: str = None, **kwargs) -> Any:
        """
        Run the configured sequence of optimization passes on the graph.

        Args:
            context: Optional OptimizationContext for telemetry and iteration control.
            debug_dir: Optional directory to save intermediate graph representations.

        Returns:
            The optimized graph representation (e.g., tf.GraphDef or torch.fx.GraphModule).
        """
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """
        Return the current number of nodes in the graph being optimized.
        Useful for reporting statistics across frameworks.
        """
        pass
