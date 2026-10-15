# tests/layers/nn/test_model_graph.cpp


Suites: `Unit_ModelGraph`, `Unit_ModelGraphDeathTest`.

Tests ModelGraph: what add and output refuse, what a trace emits, and the state names.

This file needs no device. `NN::ModelGraph` in `modules/layers/nn/include/SushiAI/nn/model_graph.hpp` holds
named nodes in the order they were added. `add` takes a name, a layer or an `NN::Node` and the
`NodeRef`s the node reads, and returns the `NodeRef` of the new node. `output` publishes a ref
under a name. `trace` returns the published values as an `NN::ModelOutputs`, and `forward`
returns the one published value. `add`, `output`, `port`, `output_ports` and `trace` return a
`Status` or a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelGraph` is a `TraceModule`,
is not a `Module`, and is detected by `Detail::HasStateWalk`.

## OpList

`OpList` is a vector of the operation of every node of a graph, in order.

## TwoPortNode

`TwoPortNode` is an `NN::Node` with one input and two ports. `forward` appends the ReLU and
then the GELU of its input. The one-argument constructor names the ports "low" and "high", and
the three-argument one takes the two names.

## BranchingModel

`BranchingModel` holds a `ModelGraph` and the refs `stem`, `up` and `mix` of three of its
nodes.

## build_branching_model

`build_branching_model` builds five nodes over an [8, 8, 3] sample and publishes nothing:
"stem", a same-padded `Conv2d(3, 4)` of kernel 3 and stride 2 on the input; "down", a
`Conv2d(4, 8)` of the same window on the stem; "up", an `Upsample2d{2}` on that; "merge", a
`Concat` of up and stem; and "mix", a pointwise `Conv2d(12, 4)` on the merge.

## ops_of

`ops_of` returns the `OpList` of a graph.

## parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields under a prefix as
"name:extents" strings.

## add_relu_declared_as

`add_relu_declared_as` adds a `LayerNode` over a `LayerAdapter<ReLU>` constructed with a given
shape. The node declares that shape on its port whatever its input is.

## RefsNameTheInputAndTheNodesInTheOrderAdded

`model.input()` reports `is_input()` and port 0. The refs of stem, up and mix report the nodes
0, 2 and 4, mix reports port 0, stem and up compare unequal, and `size()` is 5.

## ShapeOfReturnsTheDeclaredPerSampleShapes

`shape_of` is [8, 8, 3] for the input, [4, 4, 4] for stem, [4, 4, 8] for up and [4, 4, 4] for
mix. Node 3, the merge, declares [4, 4, 12] on its port.

## BranchAndMergeTracesTheNodeSequenceOfTheHandWrittenTrace

The model's `forward` under an evaluation trace is compared with the same five layers traced
by hand through a `GraphBuilder` in a second graph. The two graphs hold the same operations in
the same order, and both results are [2, 4, 4, 4]. The sequence holds one K_UPSAMPLE2D and one
K_CONCAT_CHANNELS.

## MergeEmitsOneConcatChannelsReadingBothBranches

With stem, up and mix published, the traced graph holds one K_CONCAT_CHANNELS node. Its inputs
are the published value of up and then that of stem, and its output is [2, 4, 4, 12].

## TraceReturnsEveryPublishedOutputByName

With "features" and "skip" published in that order, `trace` returns two entries in that order.
`at` returns a different id for each name, and both values are [2, 4, 4, 4].

## ForwardReturnsTheSinglePublishedOutput

A model of one ReLU node whose result is published traces one K_RELU node. That node reads the
input, and its output is the id `forward` returned.

## ForwardFailsWhenTwoOutputsArePublished

With two outputs published, `forward` returns its input id and records `InvalidDescription` on
the trace with a message containing "exactly one model output", "features" and "skip". The
graph is marked refused.

## OutputShapeIsTheShapeOfTheSinglePublishedOutput

`output_shape([8, 8, 3])` is an empty `Shape` while nothing is published, [4, 4, 8] once up is
the one published output, and an empty `Shape` again once a second output is published.

## AGraphIsANodeOfAnotherGraph

A `ModelGraph` holding a `Linear{6, 8}` named "dense" is added to an outer graph as the node
"inner", with a ReLU after it. `shape_of` of the nested node is [8], the one output is
[4, 8], and the outer walk yields "inner.dense.weight:[8, 6]" and "inner.dense.bias:[8]".

## ANestedGraphThatPublishesTwoOutputsFailsTheOuterTrace

An inner graph that publishes "first" and "second" is added as a node. The outer `trace` fails
with `InvalidDescription` and a message containing "node 'inner'", "exactly one model output",
"first" and "second".

## ASequentialAndALayerStackAreNodesOfAGraph

A `Sequential{Linear{6, 8}, ReLU}` is added as "head" and a `LayerStack` of a `Linear{8, 4}`
and a GELU as "tail". `shape_of` is [8] and [4], the output is [4, 4], and the walk yields
"head.0.weight:[8, 6]", "head.0.bias:[8]", "tail.0.weight:[4, 8]" and "tail.0.bias:[4]".

## TheInputCanBePublishedAsAnOutput

A model with no node that publishes its input returns the input id from `forward` and emits no
node.

## WalkNamesStateByNodeNameInTheOrderAdded

After a trace of the branching model, the walk under "model" yields "model.stem.weight:[4, 3,
3, 3]", "model.down.weight:[8, 3, 3, 4]" and "model.mix.weight:[4, 1, 1, 12]", and
`NN::buffer_count` is 0. The upsample and the concat contribute no name.

## ATrainingTraceReachesAModalNodeAndTheWalkNamesItsBuffers

A model of one `BatchNorm2d(3)` node named "bn", traced under a training trace, emits one
K_BATCHNORM2D node and leaves two updates in the list. The parameter walk under "model" yields
"model.bn.weight:[3]" and "model.bn.bias:[3]", and `for_each_buffer` yields
"model.bn.running_mean" and "model.bn.running_var".

## AChainNamedByIndexYieldsTheNamesOfALayerStack

A model whose nodes are named "0", "1" and "2" holds a `Linear{6, 8}`, a GELU and a
`Linear{8, 4}`. A `LayerStack` holds the same three layers. After each is traced into its own
graph, the two parameter walks are equal, the two graphs hold the same operation sequence, and
the names are "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]" and "2.bias:[4]".

## NestsInASequentialUnderItsIndex

A `Sequential<ModelGraph, ReLU>` traces to a [4, 8] value whose last node is a K_RELU. Its walk
yields "0.dense.weight:[8, 6]" and "0.dense.bias:[8]".

## AddRejectsADuplicateName

A second node named "act" fails with `InvalidDescription` and a message containing "already
has a node named 'act'" and "index 0". `size()` stays 1.

## AddRejectsAnEmptyName

The name "" fails with "needs a name" and `size()` stays 0.

## AddRejectsANameWithADot

The name "a.conv" fails with "'a.conv'" and "contains '.'" through the layer overload, and
"b.act" fails the same way through the `NN::Node` overload. `size()` stays 0.

## AddReportsARefusedNameBeforeTheLayerSeesItsSource

A `Conv2d` added under a taken name, or under an empty one, on a [6] source fails with the
message about the name. The convolution would refuse a [6] source, and that failure is not the
one returned. `size()` stays 1.

## AddReturnsTheFailureOfALayerThatRefusesItsSource

`add("stem", Conv2d::same_padded(3, 4, 3, 2), input)` on a [6] sample fails with a message
containing "node 'stem'", "Conv2d(3, 4)" and "[6]", and `size()` stays 0.

## AddRejectsAnUnknownSource

A ref to node 1 of another graph, used as the source in a graph of one node, fails with a
message containing "node 'late'", "source node 1" and "holds 1 nodes". The `NN::Node` overload
fails with "node 'later'" and "source node 1". `size()` stays 1.

## AddRejectsASourcePortTheNodeDoesNotHave

A ref to port 1 of a two-port node of another graph, used on a graph whose node 0 has one
port, fails with a message containing "node 'late'", "'only' has 1 ports" and "port 1".

## PortSelectsANamedPortThatALaterNodeReads

`port(pair, "high")` returns a ref to node 0, port 1, and `port(high, "low")` returns a ref
equal to `pair`. A ReLU added on `high` and published traces three nodes. Node 1 is the GELU,
and node 2 reads node 1's output.

## PortRejectsAnUnknownNameAndTheInput

`port(pair, "middle")` fails with a message containing "node 'pair'", "'middle'" and "low,
high". `port(input, "low")` fails with "the model input has no port named 'low'".

## OutputPortsPublishesEveryPortUnderItsOwnName

`output_ports(pair)` publishes two outputs. `at("low")` is the output of graph node 0 and
`at("high")` that of graph node 1.

## OutputPortsPublishesNothingWhenAPortNameIsTaken

With "high" already published, `output_ports(pair)` fails with "port 'high' of node 'pair'",
and `output_ports(input)` fails with "the model input has no ports to publish". A trace then
returns one output, so the failed call did not publish "low" either.

## OutputRejectsATakenNameAnEmptyNameAndAnUnknownSource

A second `output("echo", ...)` fails with "already publishes an output named 'echo'". The name
"" fails with "needs a name". A ref to a node of another graph fails with "output 'far'",
"source node 0" and "holds 0 nodes".

## TraceRejectsADeclaredShapeThatDisagreesWithTheTracedOne

A node that declares [5] and traces a ReLU of a [4, 6] value fails the trace with
`InvalidDescription` and a message containing "node 'narrow'", "port 'output'", "[5]" and
"[4, 6]".

## TraceRejectsAnInputOfAnotherSampleShape

A model over a [6] sample traced with a [4, 7] value fails with "the model input", "[6]" and
"[4, 7]". Traced with a [6] value, which has no batch axis, it fails with "the model input"
and "[6]". The graph gains no node.

## TraceRejectsAModelThatPublishesNoOutput

A model with one node and no published output fails the trace with "publishes no output", and
the graph gains no node.

## AddRejectsASourceCountTheNodeDoesNotRead

A `Concat` built over two shapes and added with one source fails with a message containing
"node 'merge'", "reads 2 values" and "given is 1". A `TwoPortNode` added with two sources fails
with "node 'split'", "reads 1 values" and "given is 2". `size()` stays 0.

## AMovedGraphKeepsItsRefsAndLeavesAnEmptyUsableGraph

A graph move-constructed from the branching model has `size()` 5, resolves the ref `up` to
[4, 4, 8] and accepts `output("features", mix)`. The graph moved from has `size()` 0. A ReLU
added to it and published traces one node whose output is the published value.

## AccessorsAreFatalOutOfRange

The case is in the suite `Unit_ModelGraphDeathTest`. On an empty graph, `shape_of` of a ref to
another graph's node stops the process with "shape_of" and "source node 0", and `node(0)`
stops it with "ModelGraph::node" and "node 0".

## AddIsFatalOnANullNode

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a null `unique_ptr<NN::Node>`
stops the process with a report containing "null node" and "'ghost'".

## AddIsFatalOnANodeWithTwoPortsOfOneName

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a `TwoPortNode` whose ports
are both named "low" stops the process with a report containing "'pair'" and "two ports named
'low'".

## TraceIncrementalHandsEachLayerThePerSampleShapeItReads

A model of the sample shape [5, 6] holds one `SliceProbe`. `trace_incremental` in a carrying
trace over a [1, 1, 6] slice hands the probe the shape [5, 6] and adds one GELU node, and the
output "y" has the shape [1, 1, 6].

## TraceStillChecksThePerSampleShapeTheIncrementalWalkSkips

`trace` of the same model over the [1, 1, 6] slice fails with `InvalidDescription`, "[5, 6]"
and "[1, 1, 6]".

## TraceIncrementalNamesTheNodeThatHasNoIncrementalForm

With a `ReLU` named "act" after the probe, `trace_incremental` fails with `Unsupported`,
"node 'act'" and "no incremental form".

## TraceIncrementalNeedsACarryingTrace

`trace_incremental` in an evaluation trace fails with `InvalidDescription` and
"keeps nothing between runs", and adds no node.

## ANestedGraphTracesItsIncrementalForm

With the probe's graph added as one node of an outer graph, `trace_incremental` of the outer
graph hands the probe [5, 6] and adds one GELU node.
