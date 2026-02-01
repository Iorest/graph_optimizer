## 2026-02-01 - O(1) Indexed Matching for General Passes
**Learning:** Using a general `Any()` wildcard pattern in a rewrite pass that only targets specific operations is a performance anti-pattern. It forces the `PatternMatcher` to evaluate the rewriter for EVERY node in the graph (O(N)), bypassing the O(1) op-type index.
**Action:** Always prefer registering specific `Op(op_type)` patterns. Upgraded `PatternRewritePass` to support multiple patterns to facilitate this for passes like `AlgebraicSimplify` and `ConstantFold`.
