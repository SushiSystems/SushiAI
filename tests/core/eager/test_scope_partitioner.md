# tests/core/eager/test_scope_partitioner.cpp


Suites: `Small_ScopePartitioner`.

Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.

This file needs no device. `ScopePartitioner::assign` reads a graph, the segments of its forward
trace and the provenance of its backward nodes, and returns a `Partition`.
`modules/core/eager/README.md`, under `scope_partitioner.hpp` and `partition.hpp`, states the
key rule the cases check: with `m` segments, key `k` forward, key `2m - k` backward, and `2m` for
the optimizer.

The cases fall into three groups, in file order:

- keys and region lists of traced and differentiated steps
- graphs built by hand to reach a K_LEAF origin or a graph with no backward node
- inputs that `assign` refuses

## Step

`Step` holds one traced, differentiated and partitioned training step: the graph, the segments,
the provenance, the parameters, one gradient per parameter and the partition.

## as_span

`as_span` returns a segment vector as the span `assign` takes.

## finish

`finish` seals the trace, differentiates the loss against `step.parameters` with a `Provenance`
and partitions the result. Every step builder ends with it, so the four steps differ only in
what they trace.

## linear

`linear` emits a K_MATMUL and a K_BIAS_ADD over a new weight and bias and appends both to the
parameter list. The parameter order is therefore weight, bias per layer, which is how a case
indexes `gradients`.

## unscoped_step

`unscoped_step` builds two linear layers and a loss with no scope open, which gives one segment.

## three_segment_step

`three_segment_step` builds scopes "first" and "second" with one linear layer and an activation
each, and a scope "loss". It has three segments, six leaves and seven other forward nodes.

## shared_weight_step

`shared_weight_step` declares one weight in a scope "weights", multiplies by it in "first" and
again in "second", and closes with "loss". It has four segments, and the first holds only a
K_LEAF.

## repeated_block_step

`repeated_block_step` opens a scope "block" twice in a loop, with one linear layer each time,
then "loss". The input is declared before the first scope, so the trace opens with an unnamed
segment that holds only a K_LEAF.

## TwoReLUGraph

`TwoReLUGraph` holds a graph built without differentiation: a parameter, a K_RELU in a segment
"first" and a K_RELU of that in a segment "second". `build_two_relu_graph` fills it, seals the
trace and sets `forward_node_count`, and leaves `origin` empty. A case then appends backward
nodes and writes their origins by hand, which reaches arrangements `differentiate()` does not
emit.

## is_leaf

`is_leaf` reports whether a node of a graph is a K_LEAF.

## segment_number

`segment_number` returns the number, counted from 1, of the segment that holds a forward node,
or 0 when none does. It is computed from the segment ranges and not from the partition, so a
forward key is compared against something the partitioner did not write.

## backward_keys_of

`backward_keys_of` returns the keys of the backward nodes whose origin is a non-K_LEAF node of
one segment. It skips K_LEAF origins because those take their key from their operands.

## AnUnscopedStepIsOneRegion

With one segment, every non-K_LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

## ForwardKeysFollowTraceOrder

Every non-K_LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the loop skips nodes it should compare.

## BackwardKeysMirrorForwardKeys

With three segments the optimizer key is 6, and the backward nodes whose origins lie in segments
1, 2 and 3 have keys 5, 4 and 3. Each segment must have at least one such node, so an empty
result cannot satisfy the loop.

The region list has five entries with keys 1 to 5, named "first#0", "second#0", "loss#0",
"second#0", "first#0". The first three are forward and the last two backward.

## TheLastSegmentSharesItsRegionWithItsBackward

The K_NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

## EveryOperandComesFromALowerOrEqualKey

For each of the four steps, every node's key is at least the key of the node that produces each
of its operands. Operands with no producer, or whose producer has no region, are skipped. The
check runs over every node of the graph and not over chosen ones, and each step must contribute
at least one comparison. `modules/core/eager/README.md` gives this as the condition for a
valid partition under `RegionKey`.

## AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that does not contain the case, and the message on the last assertion says so.

## ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an K_ACCUMULATE whose origin is a K_LEAF. Its two
operands are produced under keys 5 and 6, the backward keys of "second" and "first" among four
segments, and the K_ACCUMULATE has key 6. Key 7, the backward key of the segment that holds the
weight's K_LEAF, is not used for it.

## ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder

Two K_ADD nodes are appended to a `TwoReLUGraph` with the K_LEAF as their origin. One reads the
key 2 value first and the key 1 value second, the other the reverse, and both get key 2. A rule
that took the key of the first or of the last operand would give one of them key 1.

## RegionsListOnlyKeysThatHoldANode

`shared_weight_step` has four segments and optimizer key 8. Its region list has five entries
with keys 2 to 6. Key 1 belongs to the K_LEAF-only segment "weights" and key 7 to its backward
mirror, no node carries either, and neither is listed.

## ARepeatedScopeIsNamedByItsOccurrence

`repeated_block_step` has four segments and optimizer key 8. The region list has keys 2 to 6,
named "block#0", "block#1", "loss#0", "block#1", "block#0", so the backward regions carry the
names of the forward regions they mirror.

## AGraphWithNoBackwardNodeHasForwardRegionsOnly

A `TwoReLUGraph` with an empty `origin` gives its two K_RELU nodes keys 1 and 2 and optimizer
key 4. The region list holds "first#0" and "second#0", both forward. The optimizer key is `2m`
whether or not any backward node exists.

## LeavesBelongToNoRegion

In `three_segment_step` every K_LEAF has `K_NO_REGION` and every other node has a key. The count
of six leaves, two inputs and four parameters, shows that the loop reached the K_LEAF branch.

## SegmentsThatDoNotCoverTheForwardGraphThrow

Four malformed segment lists are refused, each with its own message: a list missing its last
segment ("The segments cover nodes 0 to", "forward graph has"), a list with a gap before segment
1 ("Segment 1 ('second')", "contiguous"), a list with an empty segment inserted ("Segment 1
('empty')", "holds no node"), and an empty list ("at least one segment").

## AProvenanceOfAnotherGraphThrows

A provenance with one origin too few is refused with a message containing "The provenance
describes" and "backward nodes". One whose last origin equals `forward_node_count`, the first
id that is not a forward node, is refused with "has origin" and "forward nodes".

## ABackwardNodeThatReadsAHigherKeyThrows

Two K_RELU nodes are appended to a `TwoReLUGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
throws, and the message contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

## ALeafOriginNodeThatReadsOnlyLeavesThrows

An K_ADD of two parameters is appended with a K_LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The message contains "as its origin" and "reads only
leaves".
