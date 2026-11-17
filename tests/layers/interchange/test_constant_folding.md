# tests/layers/interchange/test_constant_folding.cpp

Suites: `Small_ConstantFolding`.

Tests `Interchange::fold_constants`, in `modules/layers/interchange/include/SushiAI/interchange/constant_folding.hpp`,
on graphs built as values in the test. The expected constants are worked out by hand from the
operator definitions: Shape lists the extents, Gather picks along an axis (a negative index wraps
by the extent), Unsqueeze inserts extents of 1, Concat joins along an axis, Cast converts element
by element. `graph_ending_in_reshape` ends a graph in a Reshape so the folding has a reader to feed.

## FoldsAShapeToReshapeChain

`Shape(x)` of [1, 3, 8, 8], `Gather` of element 0, `Unsqueeze`, `Concat` with the constant [-1]
and two `Constant` nodes fold into the one constant `target` = [1, -1] of extent [2]. Six nodes
fold; the Conv and the Reshape stay in order.

## FoldsAConstantAndACastIntoTheScalesOfAResize

`Constant` ints [1, 1, 2, 2] cast to float become the real constant `scales` of extent [4] that the
Resize reads; only the Resize stays.

## KeepsEveryNodeOfAChainWithANonConstantInput

`Shape` of a tensor a Conv produces has no stated shape, so it and everything after it stay,
including the `Constant` that only the kept Gather reads. Nothing folds.

## KeepsAShapeWhoseInputHasASymbolicExtent

A `Shape` of an input whose first extent is the symbol `batch` stays.

## KeepsAResultThatAGraphOutputOrAnotherNodeReads

A `Shape` whose result is a graph output stays; so does one an `Add` also reads.

## SlicesAShapeByStartAndEnd

`Shape` with start 1 and end -1 of [1, 3, 8, 8] gives [3, 8].

## GathersAlongAnAxisOfATwoDimensionalConstant

Columns 2 and -3 (which wraps to 0) of [[1, 2, 3], [4, 5, 6]] on axis 1 give [[3, 1], [6, 4]].

## ConcatenatesAlongTheInnerAxis

[[1], [2]] beside [[3], [4]] on axis 1 give [[1, 3], [2, 4]].

## ReadsInitializersAsConstants

An `Unsqueeze` over the initializer [4, 5, 6] with the axes in an initializer [0] gives [[4, 5, 6]].

## RefusesAGatherIndexOutOfRange

Index 5 over an extent of 3 fails with `InvalidDescription` naming the node.

## RefusesAConcatOffTheAxisMismatch

[2, 1] against [3, 1] on axis 1 fails with `InvalidDescription` naming the node.
