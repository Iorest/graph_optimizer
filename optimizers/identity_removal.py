from ..core import Any, PassRegistry, PatternRewritePass, Op
from ..utils import create_node
from ..utils.logger import logger as logging


@PassRegistry.register("identity_removal", opt_level=1, priority=10)
class IdentityRemovalPass(PatternRewritePass):
    """
    Remove redundant Identity nodes.
    
    Transform: Identity(Identity(x))
    Into: Identity(x)
    """

    def __init__(self):
        pattern = Op(
            "Identity",
            Op("Identity", Any(alias="x"), alias="inner"),
            alias="root",
        )
        super().__init__(
            pattern, 
            self._remove_identity, 
            name="IdentityRemoval",
            optimizer_alias="identity_rm"
        )

    def _remove_identity(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        logging.info(f"[IdentityRemoval] Removing nested Identity: {root.name}")
        new_node = create_node("Identity", root.name, inputs=[x.name], attr=root.attr)
        return [new_node]
