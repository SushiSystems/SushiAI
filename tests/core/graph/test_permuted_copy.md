# tests/core/graph/test_permuted_copy.cpp

Suites: `Small_AxisPermutation`, `Small_AxisPermutationRows`, `Small_PermutedCopyPlan` and `Small_PermutedCopyPlanRows`.

Tests the checks and the plan of `permuted_copy.hpp`, the copy that permutes the non-batch axes of a
contiguous value. The file needs no device and no engine. The plans are worked by hand from the
strides of the operand: the result order names an axis per position, an axis of extent 1 is
dropped, two operand neighbours that stay neighbours join into one run, and the runs are padded in
front to four axes, each pad as wide as everything behind it.

## OnlyAListThatHoldsEachAxisOnceIsAPermutation

One row per list: the empty list, the identity, a swap, a rotation, a repeated axis, an axis beyond
the length and a negative axis. Only the first four are permutations.

## AListAsLongAsTheRankLimitIsRefused

A list with `K_MAX_RANK` entries is refused, since the batch axis takes one of the axes; one with
`K_MAX_RANK - 1` entries is accepted.

## TheInverseUndoesThePermutation, TheInverseOfARotationIsTheRotationTheOtherWay

For each of four permutations, `permutation[inverse[axis]]` is `axis`. The inverse of the rotation
2, 0, 1 is 1, 2, 0.

## ANodeHoldsThePermutationInItsFirstSlots

`permutation_of` returns slot k of a node's attributes as entry k.

## TheAxesAreTheRunsTheResultOrderWalks

Four operands: a swap of two axes of a `[2, 3, 4]` value gives the runs (2, 12), (4, 1), (3, 4)
behind a pad of 24; the identity joins everything into one run of 24; an axis of extent 1 is
dropped, which joins the batch with the axis of 5 into one run of 10; a value of one element copies
as one axis of extent 1.

## AReversalOfFiveAxesDoesNotFitInFourRuns

A `[2, 3, 4, 5, 6, 7]` value with its five axes reversed has six runs that never join, so `fits` is
false.
