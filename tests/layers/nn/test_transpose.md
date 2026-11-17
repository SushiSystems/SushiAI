# tests/layers/nn/test_transpose.cpp

That the transpose row gives the per-sample axes in the order of its permutation, moves every element to the place index arithmetic names, refuses what is no permutation, and has the gradient central differences give.

Suites: `Small_TransposeShape`, `Small_TransposeRefusal`, `Small_Transpose`, `Small_TransposeValues`, `Medium_TransposeGradient`.

Tests `NN::Transpose` and `NN::make_transpose_node` in
`modules/layers/nn/include/SushiAI/nn/transpose.hpp`. The layer traces one `K_TRANSPOSE` node;
the values cases evaluate that graph on the host in double and compare each element with the
element index arithmetic reads from the input, exactly.

## Small_TransposeShape.TheFactoryGivesThePermutedShapeAsThePort

The node has one port, named "output", whose extents are those of the input in the order of the
permutation. The rows are a swap of [5, 3], the identity of [7], the rotation [2, 0, 1], the swap
[0, 2, 1] and the reversal [2, 1, 0] of [2, 3, 4].

## Small_TransposeRefusal.TheFactoryRefusesTheRow

`InvalidDescription` with the context "layer 4 'transpose'" and a fragment per row: too few axes, a
repeated axis, an axis out of range and a negative axis ("no permutation"); no "perm" key; a
"perm" that is no array; an entry that is no integer ("perm[1]"); an unknown key ("'axis'").

## Small_Transpose.TheFactoryRefusesAnInputOfRankZero

An empty input shape fails with the context and "rank 1 or more".

## Small_TransposeValues.TheValuesFollowThePermutation

Over a [2, 2, 3, 4] input holding 0 to 47, each of the permutations [1, 0, 2], [0, 2, 1],
[1, 2, 0], [2, 0, 1] and [2, 1, 0] of the three per-sample axes gives, element for element,
what index arithmetic gives. The batch axis stays first.

## Small_Transpose.AnAxisOfExtentOneKeepsTheElementsInOrder

[2, 1, 4] with the permutation [1, 0] gives [2, 4, 1] with the elements in their order.

## Small_Transpose.TheIdentityPermutationEmitsNothing

The identity permutation returns its input value and adds no `K_TRANSPOSE` node to the graph.

## Small_Transpose.APermutationIsOneTransposeNodeHoldingIt

Swapping the first two per-sample axes of [2, 2, 3, 2] with [1, 0, 2] emits one `K_TRANSPOSE` node
and no `K_MATMUL`. The node holds 1, 0 and 2 in attribute slots 0, 1 and 2.

## Small_Transpose.AHalfPrecisionInputIsTraced

A half-precision [2, 3, 4] input with [1, 0] traces without a failure into a half-precision value of
the shape [2, 4, 3].

## Small_Transpose.AWrongLengthPermutationFailsTheTrace

A permutation of three entries over a rank-2 sample returns the input and records
`InvalidDescription` naming "no permutation".

## Medium_TransposeGradient.TheGradientOfTheSwapMatchesCentralDifferences

The gradient of a loss over a [2, 2, 3, 2] input swapped with [1, 0, 2] and reshaped to [2, 12]
matches central differences in FLOAT64 at a point whose 24 elements differ.
