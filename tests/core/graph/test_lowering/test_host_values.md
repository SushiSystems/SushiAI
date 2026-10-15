# tests/core/graph/test_lowering/test_host_values.cpp


Suites: `Medium_LoweredOperations`.

This file runs fourteen lowerings through a compiled step and compares the result with values
computed on the host: twelve forward cases on FLOAT32 and two gradient cases on FLOAT64. Each case
builds a graph with one operation, compiles it with `CompiledStep::compile`, runs it, and reads the
result back. The gradient cases reach the gradient through `Autograd::differentiate`, on a
negative log-likelihood root, and compare it with the closed-form derivative.

The forward cases allow an absolute error of 1e-6 and a relative error of 1e-5. The gradient cases
allow 1e-12 and 1e-9. These bounds are estimates, and nothing has checked them against the device
kernels.

Before this file, the integration and regression tiers built none of these operations. The unit
tier builds them as nodes and never compiles them. The fixture is a local copy of `OneShot` from
`test_forward.cpp`.

## DivDividesElementwise

DIV writes the element-wise quotient of two FLOAT32 rows.

## MinimumTakesTheSmallerOperand

MINIMUM writes the smaller of each pair of elements.

## MaximumTakesTheLargerOperand

MAXIMUM writes the larger of each pair of elements.

## AbsReturnsTheMagnitude

ABS writes the magnitude of each element.

## AtanReturnsTheArctangent

ATAN writes the arctangent of each element.

## LessEqualWritesOneForLessAndForEqual

LESS_EQUAL writes 1 where the left operand is less than or equal to the right, so the pairs that
are equal count, and 0 elsewhere.

## SelectTakesEachBranchWhereTheMaskSaysSo

SELECT takes the first branch where the mask is 1 and the second branch where it is 0.

## ConstantFillsEveryElementWithItsValue

CONSTANT fills a tensor with its value, and CONSTANT_LIKE fills one with the shape of another
operand. The step runs twice, and the outputs are checked after each run.

## SiLUMatchesItsDefinition

SILU writes x * sigmoid(x) for each element.

## SoftplusMatchesItsDefinition

SOFTPLUS writes log(1 + exp(x)) for each element.

## SiLUBackwardMatchesTheClosedFormDerivative

SILU_BACKWARD, reached through `Autograd::differentiate` on a FLOAT64 graph, matches the closed-form
derivative of SiLU.

## SoftplusBackwardMatchesTheClosedFormDerivative

SOFTPLUS_BACKWARD, reached through `Autograd::differentiate` on a FLOAT64 graph, matches the
closed-form derivative of softplus.

## StopGradientCopiesItsOperand

STOP_GRADIENT copies its operand forward unchanged. The case checks only that forward copy. It does
not check that the gradient is blocked on the device.

## BatchNormFoldWritesTheScaleAndTheShift

BATCHNORM_FOLD writes `scale = gamma / sqrt(variance + epsilon)` and
`shift = beta - mean * scale`. One of the variances is zero, so epsilon alone sets that element's
denominator.

## DepthwiseConv2dMatchesTheWindowSums

A 3x3 depthwise convolution with a pad of 1 runs over a `[1, 4, 4, 2]` image of integer values with
integer taps. Each output equals the sum of the taps whose input pixel lies inside the image,
computed by a host loop, within 1e-6.

Suites: `Small_ArgmaxLifetime`.

K_MAXPOOL2D's argmax buffer has to survive from the forward node until K_MAXPOOL2D_BACKWARD reads it,
and no test pinned that before this file.

The gradchecks cover pooling, but they cannot see this defect. If the planner handed the argmax's
bytes to a later activation, the backward would read whatever that activation wrote and route every
gradient to a plausible wrong pixel. The gradchecks would keep passing until the planner changed,
and then fail somewhere else with no mention of pooling.

The test therefore asserts on the plan and not on a number. Reuse in a MemoryPlan is byte overlap
and nothing else: two values share storage exactly when their intervals in one arena intersect. An
argmax whose interval intersects no other buffer's owns its bytes for the whole step, and that is
what "spans forward to backward" means. The check reads the plan's own arena and placement tables.
It does not read a lifetime enum, because the enum is the input to the planner's decision and the
test is about the output.

## ThePlannerKeepsMaxPoolsIndicesAliveUntilItsBackward

The graph is a pool feeding a chain long enough that the planner has real candidates to reuse the
argmax's bytes for. On a two-node graph the overlap assertion would hold because there was nothing
to overlap with, which proves nothing; the non-vacuity check at the end of the test is what gives
the assertion its meaning.

`x` is declared as a parameter and not as an input. The pool's backward is only emitted when
something upstream of it requires a gradient, and an input does not. Differentiating with respect
to `x` is what puts K_MAXPOOL2D_BACKWARD in the graph.

Between the pool and its backward the test builds a spatial mean down to [2,1,1,4], a reshape to a
matrix and a small model, so several activations are born and die while the argmax waits to be
read.

`backward_reads_the_indices` checks the premise: differentiation emitted a consumer for the
indices. Without that consumer the rest of the test is a statement about a value nobody reads.

The test asserts `plan.is_packed(pooled.indices)` before it looks for overlaps. `overlaps()`
answers false for an unpacked placement, so an argmax that had been given its own allocation would
satisfy every later assertion for a reason this test does not document.

The `overlapping_neighbours` loop is the assertion the file exists for: every other buffer in the
plan must miss the argmax entirely. Aliased names for the same bytes (K_ACCUMULATE) are not reuse, so
the loop compares by owning buffer and not by value id.

`the_planner_reused_anything` is the non-vacuity check. It requires that the planner overlapped
some pair of buffers in this graph, so the missing overlap on the argmax is a decision the planner
made and not an opportunity it never had.
