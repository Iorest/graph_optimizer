"""
Base classes for PyTorch FX optimization passes.
Mirrors the core.tensorflow.passes architecture for structural symmetry.
"""

from typing import TYPE_CHECKING, Any
import torch.fx as fx

from ..passes import BaseOptimizationPass

if TYPE_CHECKING:
    from .torch_optimizer import TorchOptimizer


class TorchBasePass(BaseOptimizationPass):
    """Base class for all PyTorch FX graph optimization passes."""

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def __init__(self, name: str = None):
        """
        Initialize a pass.

        Args:
            name: Human-readable pass name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    def apply(self, optimizer_or_module: Any) -> bool:
        """
        Satisfies the `BaseOptimizationPass` interface.

        Delegates to `transform()`, which is the actual entry point for subclasses.
        Can receive either a `TorchOptimizer` context or a raw `fx.GraphModule`.
        """
        if hasattr(optimizer_or_module, "graph_module"):
            return self.transform(optimizer_or_module.graph_module)
        return self.transform(optimizer_or_module)

    def transform(self, graph_module: fx.GraphModule) -> bool:
        """
        Execute the optimization pass.

        Args:
            graph_module: The PyTorch FX GraphModule to optimize.

        Returns:
            bool: True if the graph was modified, False otherwise.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )
