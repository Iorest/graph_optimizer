from ..core import Any, PassRegistry, PatternRewritePass, Op
from ..utils import create_node


@PassRegistry.register("identity_removal", opt_level=1, priority=10)
class IdentityRemovalPass(PatternRewritePass):
    """Pass to remove redundant Identity nodes."""

    def __init__(self):
        pattern = Op(
            "Identity",
            Op("Identity", Any(alias="x"), alias="inner"),
            alias="root",
        )
        super().__init__(pattern, self._remove_identity, name="IdentityRemoval")

    def _remove_identity(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        new_node = create_node("Identity", root.name, inputs=[x.name], attr=root.attr)
        return [new_node]
