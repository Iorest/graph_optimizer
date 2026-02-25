from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizationPass(ABC):
    """
    Abstract base class for graph optimization passes across different frameworks.

    Subclasses must implement:
    - `name`: human-readable name of the pass (e.g., "constant_fold")
    - `apply(graph)`: apply the pass to a framework-specific graph and return True if changes were made
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the pass."""
        pass

    @abstractmethod
    def apply(self, graph: Any) -> bool:
        """
        Apply this optimization pass to the given graph.

        Args:
            graph: A framework-specific graph object.
                   For TensorFlow: tf.GraphDef / TFGraphOptimizer
                   For PyTorch FX: torch.fx.GraphModule

        Returns:
            True if the graph was modified, False if it converged unchanged.
        """
        pass
