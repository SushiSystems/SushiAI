# tests/layers/nn/test_node_registry.cpp


Suites: `Small_NodeRegistry`.

Tests `NN::NodeRegistry`, `NN::register_layers`, `NN::make_node` and `NN::build_model` over
the rows `register_layers` adds. The file needs no device. `make_node` looks a type up among
the registry's node rows and then among its layer rows, and wraps a layer in an
`NN::LayerNode`. `build_model` turns a `Config::ModelSpec` into an `NN::ModelGraph`. Each
returns a `Result`. `MinimalCase`, `TraceSummary`, `span_of`, `make_minimal` and
`trace_summary` come from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`.

## K_NORM

`K_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and the momentum the defaults of
`K_BRANCHING_MODEL` state.

## MinimalCase

`MinimalCase` holds a node registry key, a row's attributes as JSON, the per-sample shapes the
row reads and the per-sample shape of the node's one port.

## K_MINIMAL_CASES

`K_MINIMAL_CASES` has one `MinimalCase` per node row of `registered()`, in row order:
batchnorm2d, concat, add, embedding, layernorm and dropout.

## LayerCase

`LayerCase` holds a layer registry key, a row's attributes as JSON and the per-sample shape
the row reads.

## K_LAYER_CASES

`K_LAYER_CASES` has one `LayerCase` per layer row of `registered()`, in row order.

## K_INPUT_JOINING_MODEL

`K_INPUT_JOINING_MODEL` is a version 2 model over [8, 8, 3] with three rows: a `conv2d` of 4
filters and kernel 1, a `silu`, and a `concat` whose `from` is `["input", -1]`.

## K_BRANCHING_MODEL

`K_BRANCHING_MODEL` is a version 2 model over [16, 16, 3] with a `defaults` block that gives
every `batchnorm2d` row `bn_epsilon` 0.001 and `bn_momentum` 0.03. Its seven rows are a
`conv2d` of 8 filters, kernel 3 and stride 2; a `conv2d` of 16 filters with the same window;
an `upsample` of scale 2; a `concat` whose `from` is `[-1, 0]`; a `conv2d` of 8 filters and
kernel 1; a `batchnorm2d`; and a `silu`.

## TraceSummary

`TraceSummary` holds the operation list of one traced graph, each published output as "name
[extents]" and each piece of state as "name [extents] parameter" or "name [extents] buffer".

## span_of

`span_of` views a vector of shapes as the `NN::ShapeSpan` of one row.

## make_minimal

`make_minimal` calls `NN::make_node` for a `MinimalCase` with given attributes and the context
"layer 0 '<type>'".

## trace_summary

`trace_summary` traces a model under a training trace over a batch of 2 samples and returns
its `TraceSummary`.

## registered

`registered` returns a registry that `register_layers` filled.

## build_branching_model_by_hand

`build_branching_model_by_hand` builds the graph `K_BRANCHING_MODEL` describes through
`ModelGraph::add`, with the nodes named "0" to "6", and publishes the last node through
`output_ports`.

## named_output

`named_output` returns an `OutputSpec` that publishes the port `NN::K_SINGLE_PORT_NAME` of a
row under a name.

## RegisterLayersAddsTheRowsInOrder

`register_layers` gives the layer rows linear, gelu, relu, tanh, sigmoid, leaky_relu,
unflatten, conv2d, maxpool2d, avgpool2d, global_avgpool, flatten, upsample, silu, mish and
slice_channels, and the node rows batchnorm2d, concat, add, embedding, layernorm and dropout,
each list in that order.

## ATypeAddedTwiceIsRefused

Adding a node row named "concat", and one named "relu", each fail with `InvalidDescription`,
"the type '<name>'" and "already". The node and layer row counts do not change.

## AnEmptyRegistryKnowsNoType

`make_node` of "silu" over an empty registry fails with `InvalidDescription`, "layer 0",
"the type 'silu'" and "does not have", and its message ends with
"the node types are: ; the layer types are: ".

## EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `registered()` has node rows, and entry `i` names
the type of row `i`.

## EachRowBuildsFromAMinimalObject

For every case, `make_node` returns a node with one port named `NN::K_SINGLE_PORT_NAME` whose
shape is the one the case lists.

## EachRowRejectsAnUnknownKey

For every case, the attributes `{"no_such_key": 1}` fail with `InvalidDescription` and a
message containing the row's context and "'no_such_key'".

## AConcatRowNamesItselfWhenItsInputsDisagree

A `concat` row over [4, 4, 2] and [8, 4, 2] fails with a message containing "layer 7
'concat'", "[8, 4, 2]" and "H and W".

## ANodeTypeBuildsItsOwnNode

`make_node("concat", ...)` over [4, 4, 2] and [4, 4, 3] returns an `NN::Concat` and not a
`LayerNode`.

## ALayerTypeIsWrappedInALayerNode

`make_node("upsample", ...)` with scale 2 over [4, 4, 2] returns a `LayerNode` with one port
named `K_SINGLE_PORT_NAME` of shape [8, 8, 2]. The case then walks `K_LAYER_CASES`, which has
one entry per layer row of the registry in order. Every type builds a `LayerNode`, and every
type given `{"no_such_key": 1}` fails with "'no_such_key'", which shows the layer's own
factory ran.

## ALayerTypeRefusesTwoInputs

A `relu` row over two shapes fails with a message containing "layer 3 'relu'", "reads 2
values" and "reads one".

## ALayerFactorysRefusalComesBackAsAStatus

A `linear` row with no attribute fails with a message containing "layer 4 'linear'" and
"'units'". A `linear` row over [8, 8, 2] fails with the context, "[8, 8, 2]" and "'flatten'".
The layer factories throw, and `make_node` returns each throw as an `InvalidDescription`
failure.

## AMaxPoolRowReadsItsTrailingPadding

Over [13, 13, 2] with `kernel` 2 and `stride` 1, the maxpool2d row gives a port of
[13, 13, 2] for `trailing_padding` 1, [13, 12, 2] for [1, 0] and [12, 12, 2] without the key.

## AnAvgPoolRowRefusesTrailingPadding

The avgpool2d row with `trailing_padding` fails with `InvalidDescription`,
"layer 0 'avgpool2d'" and "'trailing_padding'".

## AMaxPoolRowRefusesANegativeTrailingPadding

The maxpool2d row with `trailing_padding` -1 fails with `InvalidDescription`,
"layer 0 'maxpool2d'", "non-negative trailing padding" and "-1,-1".

## ASliceChannelsRowReadsItsOffsetAndCount

Over [4, 4, 64], the slice_channels row with `offset` 32 and `count` 32 gives a port of
[4, 4, 32]. With `offset` 48 and `count` 32 it fails with `InvalidDescription`,
"layer 0 'slice_channels'", "offset=48" and "count=32", and without `count` with "count".

## AnUnknownTypeListsBothTables

`make_node("conv3d", ...)` fails with a message containing "layer 5 'conv3d'", "the type
'conv3d'", "does not have", "node types" and "layer types". The message also contains every
type of the registry's node rows and of its layer rows in quotes.

## AFileAndAHandBuiltGraphAgree

`build_model` of `K_BRANCHING_MODEL` and `build_branching_model_by_hand` both hold seven
nodes. Their training traces have equal operation lists, equal outputs and equal state
records. The one output is "output [2, 8, 8, 8]". The state holds 7 records, from
"model.0.weight [8, 3, 3, 3] parameter" to "model.5.running_var [8] buffer".

## ARowReadsItsSourcesInFileOrder

In the model built from `K_BRANCHING_MODEL`, node 3 declares [8, 8, 24]: row 2's 16 channels
and then row 0's 8. Node 2 declares [8, 8, 16] and node 1 [4, 4, 16].

## OutputsArePublishedUnderTheirStatedNames

With the outputs set to row 4 as "mixed" and row 0 as "stem", the trace publishes "mixed
[2, 8, 8, 8]" and "stem [2, 8, 8, 8]" in that order.

## AnOutputWithoutANameKeepsThePortsName

An output of row 1 with an empty name is published as "output [2, 4, 4, 16]".

## ASourceMayNameAPort

A source that names the port `K_SINGLE_PORT_NAME` of the row above builds. A source that names
"no_such_port" fails with "layer 1 'conv2d'" and "'no_such_port'".

## ALaterRowMayReadTheModelInput

Row 2 of `K_INPUT_JOINING_MODEL` parses with `Config::K_MODEL_INPUT` as its first source. The
built node 2 declares [8, 8, 7]: the input's 3 channels and then row 1's 4.

## ASourceCannotNameAPortOfTheModelInput

A source that is the model input and names a port fails with a message containing "layer 2
'concat'", "the model input has no port" and "'output'".

## ARefusedRowNamesItsIndexAndType

Row 4 with `filters` 0 fails with a message containing "layer 4 'conv2d'",
"positive channel counts" and "out=0".

## ARowThatReadsARowNotAboveItIsRefused

The specs here are changed after parsing, so they hold what the parser would have refused. A
source of row 3 on row 1 fails with "layer 1 'conv2d'", "row 3" and "not above". A source of -2
on row 2 fails with "layer 2 'upsample'" and "row -2". A row with no source fails with "layer 2
'upsample'" and "no source".

## AnEmptyOutputListPublishesTheLastRowWhole

A spec whose `outputs` were cleared builds a model whose `output_shape` is [8, 8, 8], the same
as that of the model built from the parsed spec.

## AWrongOutputListIsRefused

A spec with no layer fails with "no layers". An output of row 7 fails with "output 0", "row 7"
and "rows 0 to 6". A named output with an empty port fails with "output 0", "'mixed'" and
"without a port". An output of the port "no_such_port" fails with "output 0" and
"'no_such_port'". Two outputs named "mixed" fail with "output 1" and "'mixed'".
