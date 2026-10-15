# tests/core/autograd/test_provenance.cpp


Suites: `Unit_Provenance`.

Tests the forward origin that differentiation records for every node it appends.

This file needs no device. `Autograd::differentiate()` fills a `Provenance` while it appends
backward nodes, so every case differentiates a host-side graph and reads the record back.
`modules/core/eager/README.md`, under `autograd/provenance.hpp`, states the origin rule and
what the partitioner needs it for.

## TwoLinearGraph

`TwoLinearGraph` is a two-layer perceptron with a cross-entropy loss: an [8, 6] input, a 6-to-4
linear layer, a K_RELU, a 4-to-3 linear layer, then K_LOG_SOFTMAX and K_NLL_LOSS. It keeps the ids the
cases name: the four parameters, the result of the first K_MATMUL and the loss.

## differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. It passes no seed factor and hands the `Provenance` pointer through unchanged,
which lets one case pass null.

## nodes_of

`nodes_of` returns the ids of the nodes that run one operation, starting at a given node index.
A case passes `forward_node_count` as the start, so the search covers the appended nodes only.

## EveryBackwardNodeHasAForwardOrigin

`forward_node_count` equals the node count taken before differentiation, `origin` holds one
entry per appended node, and every entry is below `forward_node_count`. An origin at or above
that count would name a backward node, which no segment of the forward trace holds.

## ARuleEmitsUnderTheNodeItDifferentiates

The node that produces the first weight's gradient is an appended node, and its origin is the
K_MATMUL that produces `first_product`. The case asserts the operation and the node id, so an
origin that named another K_MATMUL of the graph fails it.

## AFanOutAccumulateBelongsToTheProducer

`shared` is a K_RELU result that a K_TANH and a K_SIGMOID both read, so its gradient has two
contributors. The backward nodes hold exactly one K_ACCUMULATE, and its origin is the K_RELU that
produces `shared`. `x` is read once, so no closing sum is emitted and the count stays at one.

## AParameterReadTwiceAccumulatesUnderItsLeaf

The parameter `x` is read by a K_RELU and a K_TANH. The node that produces its gradient is an
K_ACCUMULATE, and the origin of that node is the K_LEAF of `x`.

`targets` is declared before `x` so that the K_LEAF of `x` is not node 0. The reverse walk ends at
node 0, so with `x` there an origin taken from the walk's last position would equal the K_LEAF's
id and the `EXPECT_NE(origin, 0u)` line could not tell the two apart.

## AReusedProvenanceHoldsOnlyTheLatestRun

One `Provenance` is filled by a first differentiation and then passed to a second one over a
`TwoLinearGraph`. Its `forward_node_count` and `origin` equal those of a fresh `Provenance`
filled from an identical graph. A `differentiate()` that appended to `origin` and did not reset
it would leave the first run's entries in front.

## ARequestedValueWithoutAProducerIsRefused

`x` is added with `add_value` and so has no K_LEAF node, and it is read twice, so the closing sum
of its gradient needs an origin that does not exist. The call throws, and the message contains
"provenance" and "producer node".

## PassingNoProvenanceChangesNothing

Two identical graphs are differentiated, one with a `Provenance` and one with null. The returned
gradient ids are equal, the node counts are equal, and every node has the same operation,
inputs and outputs. Recording is therefore an observation of the transform and not a branch in
it.
