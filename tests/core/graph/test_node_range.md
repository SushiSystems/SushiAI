# tests/core/graph/test_node_range.cpp


Suites: `Small_NodeRange`.

Tests the two fields of `NodeRange`. The header declares only the struct; compiling a step over
a range, and the refusal of a range outside the graph, are covered in
`tests/core/graph/test_compiled_step.cpp`.

## TheDefaultRangeIsTheWholeGraph

A default range starts at node 0 and ends at `K_NO_NODE`, which stands for the node count.

## ARangeKeepsTheEndsItIsGiven

A range written as {2, 5} holds 2 as its first node and 5 as its end.
