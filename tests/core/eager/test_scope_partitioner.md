# tests/core/eager/test_scope_partitioner.cpp


Suites: `Small_ScopePartitioner`, `Small_ScopePartitionerOperands`, `Medium_ScopePartitionerFatal`.

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

## The expect helpers

The loops of the file live in helpers named `expect_*`, each over a whole step so a case reads
straight-line. `expect_every_non_leaf_node_has_key`, `expect_forward_keys_follow_segments`,
`expect_backward_keys_mirror_segments`, `expect_regions_mirror_segments`, `expect_region`,
`expect_operands_come_from_lower_or_equal_keys` and `expect_leaves_hold_no_region` compare a
partition with what the segments and the graph say. `contributor_keys`, `reads_from_region`,
`count_regions_with_key` and `direction_of_region_with_key` read a partition and return a value
for the case to compare.

## AnUnscopedStepIsOneRegion

With one segment, every non-K_LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

## ForwardKeysFollowTraceOrder

Every non-K_LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the helper skips nodes it should compare.

## BackwardKeysMirrorForwardKeys

With `m` segments, the backward nodes whose origins lie in segment `k` have key `2m - k`. Each
segment must have at least one such node, so an empty result cannot satisfy the helper. The case
states the relation and not the key numbers.

## RegionsMirrorTheSegments

The region list holds the forward region of each segment in trace order, then the backward region
of every segment but the last in reverse order, since the last segment shares its forward
region. Keys run from 1, names are the segment names with their occurrences, and the directions
are forward then backward.

## TheLastSegmentSharesItsRegionWithItsBackward

The K_NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

## EveryOperandComesFromALowerOrEqualKey

Rows over the four steps: unscoped, three segments, shared weight and repeated block. Every
node's key is at least the key of the node that produces each of its operands. Operands with no
producer, or whose producer has no region, are skipped. The check runs over every node of the
graph and not over chosen ones, and each step must contribute at least one comparison.
`modules/core/eager/README.md` gives this as the condition for a valid partition under
`RegionKey`.

## AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that does not contain the case, and the message on the last assertion says so.

## ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an K_ACCUMULATE whose origin is a K_LEAF. Its
two operands are produced under two different keys, those of the backward regions of "second"
and "first", and the K_ACCUMULATE takes the higher of the two.

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
of six leaves, two inputs and four parameters, shows that the helper reached the K_LEAF branch.

## SegmentsThatStopShortOfTheForwardGraphAreFatal

A list missing its last segment stops the process ("The segments cover nodes 0 to", "forward
graph has").

## ASegmentThatLeavesAGapIsFatal

A list with a gap before segment 1 stops the process ("Segment 1 ('second')", "contiguous").

## AnEmptySegmentIsFatal

A list with an empty segment inserted stops the process ("Segment 1 ('empty')", "holds no node").

## NoSegmentAtAllIsFatal

An empty list stops the process ("at least one segment").

## AProvenanceWithTooFewOriginsIsFatal

A provenance with one origin too few stops the process with a report containing "The provenance
describes" and "backward nodes".

## AnOriginOutsideTheForwardRangeIsFatal

A provenance whose last origin equals `forward_node_count`, the first id that is not a forward
node, stops the process with "has origin" and "forward nodes".

## ABackwardNodeThatReadsAHigherKeyIsFatal

Two K_RELU nodes are appended to a `TwoReLUGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
stops the process, and the report contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

## ALeafOriginNodeThatReadsOnlyLeavesIsFatal

An K_ADD of two parameters is appended with a K_LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The report contains "as its origin" and "reads only
leaves".
