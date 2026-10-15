# tests/layers/nn/test_host_table.cpp


`Trace::host_table`: a table is a buffer leaf listed with its values, two declarations are two leaves, and a trace without a sink, a wrong type or a wrong count is refused.

Suites: `Small_HostTable`.

Tests `NN::Trace::host_table` and the `NN::HostTableList` a trace collects into with
`collect_tables`.

## ATableIsALeafListedWithItsValues

A [2, 3] float32 table of six values adds one node. Its leaf has that shape and type, the
lifetime `Buffer` and no gradient, and `graph.state_leaves()` stays empty. The list holds one
table with that leaf and the six values, and the trace is ok.

## EveryDeclarationIsItsOwnLeaf

Two float64 tables declared on a training trace return two different leaves, and the list
holds them in the order they were declared.

## ATableThatNoStepCanFillIsRefused

`host_table` fails with `InvalidDescription` and a message naming `'angles'` three times: on a
trace without `collect_tables` ("collect_tables"), for the type int32 ("float32 or a float64")
and for two values under the shape [2, 3] ("6 elements", "2 values"). The graph gains no node
and the list stays empty.
