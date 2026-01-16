from ..core import Op, Any, Variadic, PassRegistry, PatternRewritePass
from ..utils import create_node
from ..utils.logger import logger as logging


@PassRegistry.register("concat_fusion", opt_level=3, priority=40)
class ConcatFusionPass(PatternRewritePass):
    """
    Fuses multi-level ConcatV2 operations with the same axis.
    
    Transform: ConcatV2([..., ConcatV2([a, b], axis), ...], axis)
    Into: ConcatV2([..., a, b, ...], axis)
    """

    def __init__(self):
        pattern = Op(
            "ConcatV2",
            Variadic(Any(), alias="inputs"),
            Op("Const", alias="axis"),
            alias="root",
        )
        super().__init__(
            pattern, 
            self._fuse_concat, 
            name="ConcatFusion",
            optimizer_alias="concat_fuse"
        )

    def _fuse_concat(self, match, optimizer):
        root = match.matched_nodes["root"]
        axis_node = match.matched_nodes["axis"]

        root_rank = optimizer.get_node_rank(root)
        raw_axis = optimizer.get_node_attr(axis_node, "value")
        axis_val = optimizer.canonicalize_axis(raw_axis, root_rank)

        if axis_val is None:
            return None

        new_inputs = []
        changed = False

        for input_name in root.input[:-1]:
            base_name = self.clean_input_name(input_name)
            if base_name in optimizer.nodes:
                input_node = optimizer.nodes[base_name]
                if input_node.op == "ConcatV2":
                    logging.debug(f"[ConcatFusion] Found inner ConcatV2: {input_node.name}")
                    if len(input_node.input) < 2:
                        new_inputs.append(input_name)
                        continue

                    inner_axis_name = self.clean_input_name(input_node.input[-1])
                    inner_axis_node = optimizer.nodes.get(inner_axis_name)
                    if inner_axis_node and inner_axis_node.op == "Const":
                        inner_rank = optimizer.get_node_rank(input_node)
                        raw_inner_axis = optimizer.get_node_attr(
                            inner_axis_node, "value"
                        )
                        inner_axis_val = optimizer.canonicalize_axis(
                            raw_inner_axis, inner_rank
                        )

                        if inner_axis_val == axis_val:
                            # Shape check for safety
                            root_shape = optimizer.get_node_shape(root)
                            inner_shape = optimizer.get_node_shape(input_node)

                            if root_shape and inner_shape:
                                if len(root_shape) != len(inner_shape):
                                    new_inputs.append(input_name)
                                    continue
                                match_fail = False
                                for i in range(len(root_shape)):
                                    if i != axis_val:
                                        if (
                                            root_shape[i] != inner_shape[i]
                                            and root_shape[i] != -1
                                            and inner_shape[i] != -1
                                        ):
                                            match_fail = True
                                            break
                                if match_fail:
                                    new_inputs.append(input_name)
                                    continue

                            new_inputs.extend(list(input_node.input[:-1]))
                            changed = True
                            continue
            new_inputs.append(input_name)

        if not changed:
            return None

        new_inputs.append(root.input[-1])
        new_node = create_node("ConcatV2", root.name, inputs=new_inputs, attr=root.attr)
        new_node.attr["N"].i = len(new_inputs) - 1
        logging.info(f"[ConcatFusion] Fused: {root.name} ({len(root.input)-1} -> {len(new_inputs)-1} inputs)")
        return [new_node]
