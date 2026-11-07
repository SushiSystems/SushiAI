# tests/core/graph/test_node_range.cpp


Suites: `Small_NodeRangeCheck`, `Small_NodeRange`, `Medium_NodeRange`.

Tests the default `NodeRange` and `Graph::check_node_range`, which `Lowering::record` calls to refuse a
range outside the graph. The check takes a node count and needs no graph or device. Compiling a
step over a range, and the refusal of a range outside the graph through `compile`, are covered in
`tests/core/graph/test_compiled_step.cpp`.

## TheDefaultRangeIsTheWholeGraph

A step compiled over a default `NodeRange` records as many operations as one compiled over
`{0, node_count}`, and at least one.

## ARangeIsCheckedAgainstTheNodeCountOfItsGraph

Rows, against a graph of 3 nodes: the default range, the range [0, 3) and the empty range [2, 2) lie
inside it; [0, 4), which ends behind the last node, and [1, 0), which ends before it starts, do
not and fail with "does not lie in a graph".

## TheRefusalOfARangeNamesItsBounds

The range [1, 0) over 3 nodes fails with `InvalidDescription`, naming "[1, 0)" and "3 nodes".
