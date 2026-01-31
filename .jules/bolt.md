## 2026-01-31 - [O(1) Pattern Matching vs O(N) Wildcards]
**Learning:** Using a general `Any()` wildcard pattern in a rewrite pass that only targets specific operations is a performance anti-pattern. It forces the optimizer to call the rewriter for EVERY node in the graph, resulting in O(N) complexity. By registering specific `Op()` patterns, the optimizer can use its internal indexing for O(1) matching.
**Action:** Always prefer specific `Op()` patterns over `Any()` wildcards when the set of target operations is known. Refactor `PatternRewritePass` to support multiple specific patterns if a pass targets multiple op types.

## 2026-01-31 - [Const Node Shape Discovery]
**Learning:** TensorFlow `Const` nodes often store their shape information within the `tensor` proto of the `value` attribute, rather than a top-level `shape` attribute or `_output_shapes`. Standard shape extraction utilities must explicitly check the `tensor_shape` field within the `value` attribute's tensor to avoid losing shape information for constants.
**Action:** Ensure `get_node_shape` utilities handle the `Const` value tensor as a fallback for shape discovery.
