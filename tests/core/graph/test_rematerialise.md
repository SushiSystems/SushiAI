# tests/core/graph/test_rematerialise.cpp


Suites: `Medium_Rematerialise`.

The file covers the half of the K_IM2COL rematerialisation pass that the host-lane tier cannot
reach. `Small_Rematerialise` proves the rewrite has the right shape; this file proves the
rewritten graph computes the same numbers.

Checking that the pass ran and the graph still executes would prove nothing. The pass deletes
a stored value and recomputes it. Every way that can go wrong, such as a clone reading a
different operand or a clone whose attributes were rebuilt from the geometry and not copied,
produces a column matrix of the right shape holding the wrong pixels. So the assertion is
equality against the same graph run without the pass. The equality is exact and not a
tolerance: the clone recomputes the same operation from the same inputs, and SushiRuntime's
`sushiruntime_deterministic_fp` contract makes "the same" mean bit for bit.

## image_values

The [1, 5, 4, 2] image is not square on purpose. K_IM2COL's output extents live in attribute
slots 8 and 9. A clone goes wrong by re-deriving them where it should copy them, which can
yield the transpose of the real window grid with the same number of windows. On a square image
that mistake is invisible; on 5x4 it is a different gather.

## TheRewrittenGraphComputesTheSameGradient

The node counts are asserted first, to show the pass did something. A pass that no-ops passes
the comparison that follows trivially.

The comparison is exact, not near. The clone recomputes the same operation from the same
operands in the same order.

## TheColumnMatrixLeavesTheActivationArena

The pass exists to take the column matrix out of the activation arena. The test states that
against the planner and does not infer it from the lifetime field. The planner turns the
shortened live range into an arena decision, and it has to agree.
