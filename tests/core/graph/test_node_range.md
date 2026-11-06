# tests/core/graph/test_node_range.cpp


Suites: `Medium_NodeRange`.

Tests the default `NodeRange`. The header declares only the struct; compiling a step over
a range, and the refusal of a range outside the graph, are covered in
`tests/core/graph/test_compiled_step.cpp`.

## TheDefaultRangeIsTheWholeGraph

A step compiled over a default `NodeRange` records as many operations as one compiled over
`{0, node_count}`, and at least one.
