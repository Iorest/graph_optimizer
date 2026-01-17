from ..core import Any, PassRegistry, PatternRewritePass, Op, RewriteResult
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


@PassRegistry.register("identity_bypass", opt_level=2, priority=15)
class IdentityBypassPass(PatternRewritePass):
    """
    Remove single Identity nodes by bypassing them.
    
    Transform: Identity(x) where consumers don't need the Identity
    Into: redirect consumers directly to x
    
    Note: Protected nodes (output nodes) are not bypassed.
    """

    def __init__(self):
        pattern = Op("Identity", Any(alias="input"), alias="identity")
        super().__init__(
            pattern,
            self._bypass_identity,
            name="IdentityBypass",
            optimizer_alias="identity_bypass"
        )

    def _bypass_identity(self, match, optimizer):
        identity_node = match.matched_nodes["identity"]
        input_node = match.matched_nodes["input"]
        
        # Don't remove Identity that reads from resource (needed for variable reads)
        if "ReadVariableOp" in identity_node.name:
            return None
        
        # Don't bypass if this is a protected/output node
        if hasattr(optimizer, 'protected_nodes') and identity_node.name in optimizer.protected_nodes:
            return None
        
        # Check if Identity has special attributes that might be needed
        # (e.g., _class for colocation)
        if '_class' in identity_node.attr:
            return None
        
        # Get the input name (handle control dependencies)
        input_name = identity_node.input[0] if identity_node.input else None
        if not input_name:
            return None
        
        # Create mapping from identity to its input
        logging.debug(f"[IdentityBypass] Bypassing {identity_node.name} -> {input_name}")
        
        return RewriteResult(
            new_nodes=[],  # No new nodes, just remove this one
            replaced_nodes=[identity_node.name],
            node_mapping={identity_node.name: input_name}
        )
