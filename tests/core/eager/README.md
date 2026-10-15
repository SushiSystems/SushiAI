# tests/core/eager

The tests of the module `core/eager`, one file per header of `modules/core/eager/include/`. The suite prefix of a case gives its ctest label.

## test_region_boundary.cpp

Suites: `Unit_RegionBoundary`.

Tests which values `find_boundaries` lists as the imports and exports of a region.

This file needs no device. Every case traces a small graph on the host, partitions it with
`ScopePartitioner` and reads the `BoundaryMap`. `modules/core/eager/README.md`, under
`region_boundary.hpp`, says when a buffer is shared and why a region exports every value of a
shared buffer.

### eager_test_support.hpp

The host-lane tests of the module share the traced steps of `eager_test_support.hpp`.
`encoder_head_loss(head, extra)` traces scopes "encoder", "head" and "loss" over an [8, 6]
input, differentiates the loss and partitions the result; `extra` adds a scope "extra" between
the head and the loss, which turns three segments into four. `linear_then_relu(split)` traces
`relu(linear(x))` with no backward pass, as one segment or as two. `presets_of` gives each leaf
the id `K_LEAF_STORAGE` plus its position among the leaves and each parameter gradient
`K_GRADIENT_STORAGE` plus the parameter's position, which is the storage a `Session` would
supply. The positions are the same in every variant of a step, so two traces of one model give
their leaves the same ids. `observed_of` returns the loss and the gradients, the values a session
observes.

### AForwardRegionExportsWhatALaterRegionReads

The encoder's result is read by the head, so the encoder's forward region exports it and the
head's forward region imports it. The head's result is read by the loss region.

### AnObservedValueIsExportedWithNoReader

One region holds the whole step. Its result is exported when the caller observes it and is not
exported when nobody does.

### AnExportIsNamedByItsPlaceInTheRegion

The region holds K_MATMUL, K_BIAS_ADD and K_RELU; the three K_LEAF nodes are in no region. The
observed result is therefore at node 2, result 0. The position counts the region's own nodes,
which is what keeps it stable when an earlier region grows.

### ImportsAreListedOnceInReadOrder

`add(add(b, a), add(a, b))` reads `b` first and each leaf twice. The imports are `b`, `a`.

### EveryImportIsALeafOrAnExportOfALowerKey

The case walks every region of a four-segment differentiated step. Each import that a node
produces comes from a region with a lower key, and that region lists it as an export. `checked`
counts the imports the loop compared, so a map with no boundary value would fail the last
assertion.

### AnAccumulateSharesTheBufferOfItsFirstOperand

K_ACCUMULATE writes its result into its first operand. The result and that operand have one
buffer and the second operand has its own. The result is observed, so both values of the buffer
are exports and the second operand is not.

### AnAccumulateIntoALeafThrows

An accumulate whose first operand is an input leaf would add into storage the session does not
own.

### APartitionOfAnotherGraphThrows

A partition with one node too few is refused before any node is read.

## test_region_emitter.cpp

Suites: `Unit_RegionEmitter`.

Tests that `RegionEmitter` records a region's program on the recorder it is handed.

This file needs an engine: its cases allocate tensors and run a recorded graph. The emitter is
handed the recorder of a static `SushiRuntime::Graph`, which shows that it depends on a
`Recorder` and on nothing of a dynamic graph.

### Recorded

`make_recorded` compiles `relu(x * weight + bias)` as one region, adopts the program's plan in a
`TensorPool` and binds a tensor filled by `fill_wave` to each import and export.
`expected_result` reads the three imports back and computes the result on the host with
`linear_relu`. The imports are in the order the region reads them: `x`, `weight`, `bias`.

### EmitRecordsTheProgramOnTheGraphItIsHanded

After `emit()` the graph runs once and the export holds the host result within `K_TOLERANCE`,
1e-5, the distance a FLOAT32 sum of six products keeps from the same sum in double.

### EmitCountsItsCallsAndItsTasks

Two `emit()` calls on two graphs record the same number of nodes and leave `emit_count()` at 2.
A region recorded again under a new key is this second call. The program has no dropout, so it
asks for no replay prelude.

## test_region_executor.cpp

Suites: `Unit_RegionExecutor`.

Tests that `RegionExecutor` builds, keeps, records again and drops regions as a step says.

This file needs an engine. `modules/core/eager/README.md`, under `region_executor.hpp`, gives the
two passes of `execute` and says why a kept region is not touched.

### Harness

`Harness` does a session's planning for a host-traced step: it allocates a filled tensor for
each leaf, plans the step on a `RegionTable`, allocates the created slots and compiles each
built region. `prepare` returns the `StepWork`, and `run` executes it. `expected` computes
`relu(x * weight + bias)` on the host from the leaves; the first leaf of `linear_then_relu` is
`x`, so it sits in `K_LEAF_STORAGE`.

### ABuiltStepWritesItsObservedValue

One region is built, lowered once and composed once, and the observed result matches the host.

### AKeptStepRunsAgainWithoutBeingComposedOrRecorded

The second step traces the same graph. The executor composes nothing, lowers nothing and
allocates no pool byte. The input tensor was refilled between the steps, and the result follows
it, so the kept kernels read the storage they were recorded with.

### TwoRegionsRunInKeyOrder

A scope between the linear layer and the activation gives two regions. The result equals the
one-region result, so the region with the lower key ran first.

### ARegionRecordedUnderAnotherKeyKeepsItsProgram

The case hands the executor a `RegionOrder` with the action `Record`, the key raised by 2 and
the old key as `previous_key`, which is what the table emits when a scope appears. The old key
holds no region afterwards, the new one holds a program of the same node count, the region was
lowered twice and the graph composed twice.

### ARegionThatLeftIsDropped

Two regions, then one: the step that joins them drops the activation region by its old key.

### TheTrailingWorkRunsBehindEveryRegionOnEveryRun

The update subtracts the region's result from a zeroed tensor, so that tensor holds the negated
result only if the update ran after the region. The second step hands over the same recorder
under the same key: it is replayed and not recorded again.

### DiscardDropsEveryRegionAndAKeptRegionIsThenRefused

After `discard()` the executor holds no region. The harness's table was not told, so it plans a
`Keep`, and `execute` returns `InvalidDescription`. A session calls `RegionTable::forget` with
`discard()` for this reason.

## test_region_program.cpp

Suites: `Unit_RegionProgram`.

Tests the graph, the plan and the reports `compile_region` makes for one region.

This file needs no device. `modules/core/eager/README.md`, under `region_program.hpp`, describes
the program.

### TheGraphHoldsTheRegionsNodesBehindOneLeafPerImport

The region "activation#0" holds one K_RELU and reads the linear layer's result. With fusion off
its program has two nodes: a K_LEAF that writes the import, and the K_RELU.

### ValuesKeepTheIdsOfTheStep

A value has the same id in the program and in the step, so the storage list of the step addresses
both.

### AnImportAnotherRegionComputesIsAnInputOfTheProgram

The linear layer's result is an activation in the step and has `Lifetime::Input` in the program
that imports it.

### AParameterImportKeepsItsLifetime

A leaf of the step keeps the lifetime it was declared with.

### NoSharedValueIsPackedIntoAnArena

Over every region of a differentiated three-segment step, no import and no export is packed. A
packed export would be overwritten by a later value of the same region while another region
still reads it.

### AChainInOneRegionFolds

With no scope, K_MATMUL, K_BIAS_ADD and K_RELU fold into one `GEMMBiasReLU` and the program
holds no K_RELU.

### AChainCutByAScopeFoldsOnlyUpToTheBoundary

With a scope between the layer and the activation, the first region folds a `GEMMBias`, no
region holds a `GEMMBiasReLU`, and the second region keeps its K_RELU. This is the price of a
boundary the design names.

### SwitchingFusionOffLeavesTheNodesAsTraced

`ProgramOptions::fusion` reaches the pass.

### AKeyThatNamesNoRegionThrows

`K_NO_REGION` is the key of the leaves and compiles to nothing.

## test_region_signature.cpp

Suites: `Unit_RegionSignature`.

Tests the signature that `signature_of` computes for one region of a partition.

This file needs no device. `Eager::signature_of` reads a graph, a partition and one identity per
value, all of which a case builds on the host. `modules/core/eager/README.md`, under
`region_signature.hpp`, lists what the hash covers and gives the reason for each operand rule.

Most cases build two graphs that differ in one property and compare the two signatures. Where
the operation list is the same in both, the case asserts that first, so the difference it then
asserts is a difference in the hash alone.

### Region keys and identities

`K_BODY` is key 1, the one region of a hand-built partition, and `K_READER` is key 2, the region
that reads from it in a two-region partition. `K_PARAMETER_IDENTITY` and `K_EXTERNAL_IDENTITY`
are the default identities of the parameter and of the value `K_READER` reads from outside.
They are stated once so that a case names the one it changes.

### Traced

`Traced` holds a graph, its partition and one identity per value id, which are the three things
`Eager::signature_of` reads.

### signature_of

The file-local `signature_of` has three overloads. The first takes a `Traced` and a region key
and calls `Eager::signature_of` with the identity vector as a span. The other two take a
`Recipe` or an `ExternalRecipe`, build its graph and return the signature of `K_BODY` or of
`K_READER`.

### Recipe

`Recipe` describes `scale(combine(relu(x), tanh(x)), alpha)` over a parameter `x`, and each of
its fields is one property a case varies: the shape, the element type, the K_SCALE factor, K_SUB in
place of K_ADD, the operand order of the combine, and the identity of `x`. `build(Recipe)` puts
the four operations under `K_BODY` and the K_LEAF under no region, and gives every value but `x`
identity zero.

### node_of

`node_of` returns the first node of a graph that runs a given operation. The attribute and
layout cases use it to reach the node they edit after `build` returns.

### assign_regions

`assign_regions` gives a `Traced` its partition from one key per non-K_LEAF node, in node order,
and a region list. A K_LEAF gets `K_NO_REGION`. It also resets every identity to zero, so a
builder that calls it sets afterwards only the identities its case is about.

### ExternalRecipe

`ExternalRecipe` describes a graph whose `K_READER` region holds one K_SCALE that reads a value
the region does not produce. Its fields choose whether that value is the parameter `x` or the
K_RELU result of `K_BODY`, whether `K_BODY` appears in the region list, and which identity the
value has.

### ProbeOperands

`ProbeOperands` names four operand orders for a hand-built three-operand K_ADD in `K_READER`.
`build(ProbeOperands)` leaves every identity at zero, and the one in-region operand is result 0
of the region's node 0. Both orders of a pair therefore feed the hash the same words, and the
two signatures can differ only if the kind of each operand enters the hash.

### build_reader_of_result

`build_reader_of_result` adds a K_RELU node by hand with two results and a K_TANH that reads the
result at the given index, both under `K_BODY`. The K_RELU is added through `Graph::add_node`
because `GraphBuilder::relu` returns one result.

### with_identity

`with_identity` records an identity for one value and returns the value id, so that
`ScopedPerceptron` can assign an identity in the expression that declares a leaf.

### info_of

`info_of` returns the `RegionInfo` of a key, or null when the partition does not list the key.

### role_identity

`role_identity` hashes a region's name, its direction, how many nodes of the region follow the
producing node, and the index of the value among that node's results. Counting from the
region's end keeps the identity of a region's last result the same when nodes are added before
it.

### identify_produced_values

`identify_produced_values` gives every value that a region's node writes its `role_identity`.
In the tests it stands in for the caller of `signature_of`, which owns the storage a boundary
value is bound to; `modules/core/eager/README.md` says why only the caller can supply that
identity.

### ScopedPerceptron

`ScopedPerceptron` traces three scopes, "stem", "body" and "head", then differentiates and
partitions the graph with `ScopePartitioner`. Its argument is the number of linear layers in
"stem". Leaves get fixed identities and produced values get role identities. `key_of` returns
the key of a region by name and direction, and `stem_output` is the value "body" reads from
"stem".

### TheSameTraceGivesTheSameSignature

Two graphs built from the default `Recipe` have equal hashes and compare equal. The operation
list is K_RELU, K_TANH, K_ADD, K_SCALE, which is the region's nodes in order with the K_LEAF left out.

### AChangedAttributeChangesTheHash

A K_SCALE factor of 3.0 in place of 2.0 leaves the operation list equal and makes the hash
differ.

### AChangedLastAttributeSlotChangesTheHash

The case writes 1.0 into slot `Attributes::K_CAPACITY - 1` of the K_SCALE node and expects another
hash. It covers the last slot because a hash that read a fixed number of leading slots would
agree with the other attribute case and miss this one.

### AChangedLayoutChangesTheHash

The K_RELU result is retagged NCHW in one of two otherwise equal graphs, and the hashes differ.
Shape and element type are the same in both, so the layout is the only input that moved.

### AChangedShapeChangesTheHash

A shape of [4, 5] in place of [4, 3] leaves the operation list equal and makes the hash differ.

### AChangedDTypeChangesTheHash

FLOAT64 in place of FLOAT32 leaves the operation list equal and makes the hash differ.

### AChangedOpChangesTheHash

K_SUB in place of K_ADD makes both the operation list and the hash differ. It is the one
`Recipe` case in which the two lists are unequal.

### ADifferentParameterChangesTheHash

The same structure over a parameter with identity `K_PARAMETER_IDENTITY + 1` has the same
operation list and another hash. The parameter is external to the region, so its identity is
the only thing the hash holds about it.

### AnEarlierRegionGrowingDoesNotChangeALaterRegion

`ScopedPerceptron(1)` and `ScopedPerceptron(2)` differ by one linear layer in "stem". The three
`ASSERT` lines at the top establish that the case tests what it claims: the value "body" reads
has a different id in the two graphs, the same identity in both, and the forward and backward
regions of "body" have different keys.

For each direction, the signature of "body#0" has a non-empty operation list and is equal in
the two graphs, by hash and by `operator==`. The forward signature of "stem#0" differs between
them, which shows that the two graphs are not equal throughout.

### ARewiredOperandChangesTheHash

`add(relu(x), tanh(x))` and `add(tanh(x), relu(x))` have the same operation list and different
hashes. Both operands are produced inside the region, so the case covers the in-region operand
position.

### AReadOfAnotherResultChangesTheHash

A K_TANH that reads result 1 of a two-result node hashes apart from one that reads result 0. The
producer's index in the region is the same in both, so the result index is what separates them.

### ABoundaryValueOfAnotherIdentityChangesTheHash

The K_SCALE in `K_READER` reads the K_RELU result of `K_BODY`. With identity
`K_EXTERNAL_IDENTITY + 1` for that value, the operation list is equal and the hash differs.

### ALeafAndABoundaryValueOfOneIdentityHashAlike

The reader's signature is equal, by hash and by `operator==`, whether its operand is the
parameter `x` or the result of another region, as long as both carry `K_EXTERNAL_IDENTITY`.
Where an external operand comes from in the graph does not enter the hash.

### AnOperandFromAnUnlistedRegionHashesAsExternal

The same graph is signed with `K_BODY` present in the region list and with it absent, and the
two signatures of `K_READER` are equal. An operand whose producer has a key other than the
region's is external whether or not the partition lists that key.

### AnOperandInsideTheRegionHashesApartFromOneOutside

The probe node reads one result of `K_READER` and the result of `K_BODY` twice, in the two
orders `InsideThenOutside` and `OutsideThenInside`. The operation lists are equal and the
hashes differ.

### ALeafOperandHashesApartFromOneInsideTheRegion

The same pair of orders with the leaf `x` in place of the `K_BODY` result,
`InsideThenLeaves` and `LeavesThenInside`, also gives equal operation lists and different
hashes.

### EqualHashesWithDifferentOpsCompareUnequal

Three signatures are written by hand with hash 42. The one listing K_ADD and the one listing K_SUB
compare unequal under `==` and `!=`, and two that both list K_ADD compare equal. A real collision
cannot be produced on demand, so the case constructs one.

### ARegionThePartitionDoesNotListThrows

`signature_of` throws for key 6 and for `K_NO_REGION`. Each message names the key, "Region 6"
and "Region 0", and contains "not listed".

### APartitionOfAnotherNodeCountThrows

One entry is removed from `region_of_node`, which leaves four entries for a five-node graph.
The message contains "assigns 4 nodes" and "holds 5".

### AnExternalValueWithoutAnIdentityThrows

The identity vector is cleared, so the id of `x` lies beyond its end. The message contains
"External value 0" and "no identity". The call throws and does not read past the span.

## test_region_table.cpp

Suites: `Unit_RegionTable`.

Tests what `RegionTable` keeps, records again, builds and drops from one step to the next.

This file needs no device: the table works on storage ids. Each case plans two or three traces
of `encoder_head_loss` on one table, with the presets of `eager_test_support.hpp` as the storage
of the leaves and the gradients. `modules/core/eager/README.md`, under `region_table.hpp` and
`slot_ledger.hpp`, states the rules.

### TheFirstStepBuildsEveryRegion

Every region is built, none has a previous key, and the plan creates slots.

### TheSameTraceKeepsEveryRegion

A second trace of the same model keeps every region and creates and releases nothing. The
encoder's result and the loss are in the storage the first plan gave them.

### AChangedScopeRebuildsItsTwoRegionsAndNoOther

The head ends in K_TANH in the first step and in K_RELU in the second. Its forward and backward
regions are built. The encoder's two regions and the loss region are kept: each reads the
head's results from the slots of the first step, because it names them by its own import places
and its own structure did not change.

### AReaderKeepsTheStorageOfAResultWhoseProducerChanged

The head's result has one storage id in both steps. This is the single fact the case above
rests on, asserted on its own.

### AScopeThatAppearsRenumbersARegionWithoutRebuildingIt

Three segments give the encoder's backward region key 5 and four give it key 7. The region is
recorded again under the new key with its old key as `previous_key`. The encoder's forward
region keeps key 1 and is kept, and the new scope's region is built with no previous key.

### AScopeThatDisappearsIsDroppedByItsOldKeys

The plan lists the two keys the "extra" scope's regions held, forward before backward, and
releases the slots only they used.

### ALeafWithoutStorageThrows

A plan with no preset names the region and the value it could not place.

### ForgetBuildsEveryRegionAgain

`forget()` returns one id per slot the first plan created, and the next plan builds every region
with nothing to drop.

## test_scope_partitioner.cpp

Suites: `Unit_ScopePartitioner`.

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

### Step

`Step` holds one traced, differentiated and partitioned training step: the graph, the segments,
the provenance, the parameters, one gradient per parameter and the partition.

### as_span

`as_span` returns a segment vector as the span `assign` takes.

### finish

`finish` seals the trace, differentiates the loss against `step.parameters` with a `Provenance`
and partitions the result. Every step builder ends with it, so the four steps differ only in
what they trace.

### linear

`linear` emits a K_MATMUL and a K_BIAS_ADD over a new weight and bias and appends both to the
parameter list. The parameter order is therefore weight, bias per layer, which is how a case
indexes `gradients`.

### unscoped_step

`unscoped_step` builds two linear layers and a loss with no scope open, which gives one segment.

### three_segment_step

`three_segment_step` builds scopes "first" and "second" with one linear layer and an activation
each, and a scope "loss". It has three segments, six leaves and seven other forward nodes.

### shared_weight_step

`shared_weight_step` declares one weight in a scope "weights", multiplies by it in "first" and
again in "second", and closes with "loss". It has four segments, and the first holds only a
K_LEAF.

### repeated_block_step

`repeated_block_step` opens a scope "block" twice in a loop, with one linear layer each time,
then "loss". The input is declared before the first scope, so the trace opens with an unnamed
segment that holds only a K_LEAF.

### TwoReLUGraph

`TwoReLUGraph` holds a graph built without differentiation: a parameter, a K_RELU in a segment
"first" and a K_RELU of that in a segment "second". `build_two_relu_graph` fills it, seals the
trace and sets `forward_node_count`, and leaves `origin` empty. A case then appends backward
nodes and writes their origins by hand, which reaches arrangements `differentiate()` does not
emit.

### is_leaf

`is_leaf` reports whether a node of a graph is a K_LEAF.

### segment_number

`segment_number` returns the number, counted from 1, of the segment that holds a forward node,
or 0 when none does. It is computed from the segment ranges and not from the partition, so a
forward key is compared against something the partitioner did not write.

### backward_keys_of

`backward_keys_of` returns the keys of the backward nodes whose origin is a non-K_LEAF node of
one segment. It skips K_LEAF origins because those take their key from their operands.

### AnUnscopedStepIsOneRegion

With one segment, every non-K_LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

### ForwardKeysFollowTraceOrder

Every non-K_LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the loop skips nodes it should compare.

### BackwardKeysMirrorForwardKeys

With three segments the optimizer key is 6, and the backward nodes whose origins lie in segments
1, 2 and 3 have keys 5, 4 and 3. Each segment must have at least one such node, so an empty
result cannot satisfy the loop.

The region list has five entries with keys 1 to 5, named "first#0", "second#0", "loss#0",
"second#0", "first#0". The first three are forward and the last two backward.

### TheLastSegmentSharesItsRegionWithItsBackward

The K_NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

### EveryOperandComesFromALowerOrEqualKey

For each of the four steps, every node's key is at least the key of the node that produces each
of its operands. Operands with no producer, or whose producer has no region, are skipped. The
check runs over every node of the graph and not over chosen ones, and each step must contribute
at least one comparison. `modules/core/eager/README.md` gives this as the condition for a
valid partition under `RegionKey`.

### AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that does not contain the case, and the message on the last assertion says so.

### ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an K_ACCUMULATE whose origin is a K_LEAF. Its two
operands are produced under keys 5 and 6, the backward keys of "second" and "first" among four
segments, and the K_ACCUMULATE has key 6. Key 7, the backward key of the segment that holds the
weight's K_LEAF, is not used for it.

### ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder

Two K_ADD nodes are appended to a `TwoReLUGraph` with the K_LEAF as their origin. One reads the
key 2 value first and the key 1 value second, the other the reverse, and both get key 2. A rule
that took the key of the first or of the last operand would give one of them key 1.

### RegionsListOnlyKeysThatHoldANode

`shared_weight_step` has four segments and optimizer key 8. Its region list has five entries
with keys 2 to 6. Key 1 belongs to the K_LEAF-only segment "weights" and key 7 to its backward
mirror, no node carries either, and neither is listed.

### ARepeatedScopeIsNamedByItsOccurrence

`repeated_block_step` has four segments and optimizer key 8. The region list has keys 2 to 6,
named "block#0", "block#1", "loss#0", "block#1", "block#0", so the backward regions carry the
names of the forward regions they mirror.

### AGraphWithNoBackwardNodeHasForwardRegionsOnly

A `TwoReLUGraph` with an empty `origin` gives its two K_RELU nodes keys 1 and 2 and optimizer
key 4. The region list holds "first#0" and "second#0", both forward. The optimizer key is `2m`
whether or not any backward node exists.

### LeavesBelongToNoRegion

In `three_segment_step` every K_LEAF has `K_NO_REGION` and every other node has a key. The count
of six leaves, two inputs and four parameters, shows that the loop reached the K_LEAF branch.

### SegmentsThatDoNotCoverTheForwardGraphThrow

Four malformed segment lists are refused, each with its own message: a list missing its last
segment ("The segments cover nodes 0 to", "forward graph has"), a list with a gap before segment
1 ("Segment 1 ('second')", "contiguous"), a list with an empty segment inserted ("Segment 1
('empty')", "holds no node"), and an empty list ("at least one segment").

### AProvenanceOfAnotherGraphThrows

A provenance with one origin too few is refused with a message containing "The provenance
describes" and "backward nodes". One whose last origin equals `forward_node_count`, the first
id that is not a forward node, is refused with "has origin" and "forward nodes".

### ABackwardNodeThatReadsAHigherKeyThrows

Two K_RELU nodes are appended to a `TwoReLUGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
throws, and the message contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

### ALeafOriginNodeThatReadsOnlyLeavesThrows

An K_ADD of two parameters is appended with a K_LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The message contains "as its origin" and "reads only
leaves".

## test_scope_trace.cpp

Suites: `Unit_ScopeTrace`.

Tests how ScopeTrace cuts a trace into segments and which calls it rejects.

This file needs no device. A `ScopeTrace` reads the node count of a graph, so every case
appends nodes to a host-side graph, opens and closes scopes around them and inspects what
`seal()` returns. `modules/core/eager/README.md`, under `segment.hpp` and `scope_trace.hpp`,
defines a segment and the names a scope may have.

The cases fall into two groups, in file order: the segments a trace yields, and the calls
`open()`, `close()` and `seal()` reject.

### trace_nodes

`trace_nodes` appends a given number of K_ADD nodes to a graph, each adding one input value to
itself. The input is added as a value with no K_LEAF node, so the node count rises by exactly the
number asked for and a case can state every range as a literal.

### segment_is

`segment_is` compares one segment with an expected name, occurrence, begin and end. On a
mismatch it prints all four fields of both, so a failure shows which field differs.

### NodesOutsideAnyScopeFormOneUnnamedSegment

Three nodes traced with no scope open give one segment: empty name, occurrence 0, nodes 0 to 3.

### AScopeSplitsTheTraceIntoThreeSegments

One node, then two in a scope "body", then three give three segments. The unnamed run after the
scope has occurrence 1, because the unnamed run before it was occurrence 0.

### ANestedScopeIsNamedByItsPath

Inside "outer" and "inner" the depth is 2 and the segment is named "outer/inner". The nodes of
"outer" before and after it form two segments named "outer", with occurrences 0 and 1.

### AScopeOpenedTwiceGetsRisingOccurrences

A scope "cell" opened three times in a loop, with two nodes each time, gives three segments
named "cell" with occurrences 0, 1 and 2 and consecutive ranges.

### AnEmptyScopeLeavesNoSegment

A scope "cell" that closes with no node traced leaves no segment. The next "cell", which holds
one node, has occurrence 0, so the dropped scope did not use up an occurrence.

### SealWithAScopeStillOpenThrows

`seal()` throws while "outer" and "inner" are open, and the message contains "seal()", "still
open" and the path "outer/inner". The depth is still 2 afterwards.

The case then traces one more node, closes both scopes and seals. The result is one segment
"outer/inner" over nodes 0 to 2, which shows the failed `seal()` did not cut the running
segment.

### CloseWithNoScopeOpenThrows

`close()` throws on a new trace, and again after one matched `open()` and `close()`. Both
messages contain "close()" and "no scope open".

### SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

### OpenRejectsAnEmptyName

`open("")` throws with a message containing "open()" and "empty scope name", and the depth
stays 0.

### OpenRejectsANameWithASlash

`open("outer/inner")` throws. The message contains the name and "may not contain", and the
depth stays 0.

### OpenRejectsANameWithAHash

`open("cell#0")` throws. The message contains the name and "may not contain", and the depth
stays 0.

### ARejectedNameDoesNotSplitTheRunningSegment

One node is traced, `open("a/b")` throws, and one more node is traced. `seal()` returns one
unnamed segment over nodes 0 to 2. A trace that ended the running segment before it validated
the name would return two.

### ARejectedCloseDoesNotSplitTheRunningSegment

The same arrangement with a `close()` that throws for having no scope open also returns one
unnamed segment over nodes 0 to 2.

### ASecondSealThrows

A second `seal()` throws with a message containing "seal()" and "sealed trace".

### OpenAndCloseAfterSealThrow

After `seal()`, `open("late")` and `close()` both throw with "sealed trace" in the message, and
the depth stays 0.

## test_slot_ledger.cpp

Suites: `Unit_SlotLedger`.

Tests which slot `SlotLedger` gives a shared buffer from one step to the next.

This file needs no device. A case opens a step, asks for slots and closes it, twice or three
times. `K_PRODUCER` writes the buffer and `K_READER` reads it; a reader names it by its own id
and the place of the value among its imports. `modules/core/eager/README.md`, under
`slot_ledger.hpp`, gives the three questions the ledger answers.

### match, first_step

The file-local `match` calls `match_readers` with every reader trusted and falls back to
`place`, the order `RegionTable` uses for a changed producer. `first_step` runs one step that
creates a slot.

### TheFirstMatchCreatesASlot

The slot is listed in `created()` with the extents and the element type asked for.

### AnUnchangedProducerKeepsItsSlot, KeepFindsNoSlotAtAnotherPosition

`keep` returns the slot written at the same position and nothing for another position.

### AChangedProducerReusesTheSlotItsReaderRead

The producer grew and the value moved from node 2 to node 5. The reader still names it as its
first import, so the buffer takes the old slot. The third step finds it by `keep` at the new
position, which shows that a matched slot is filed under its new name.

### AReaderThatIsNotTrustedNamesNoSlot

With no trusted reader `match_readers` returns `K_NO_STORAGE`, although a reader of that name
read the slot in the step before.

### PlaceTakesTheSlotWrittenAtTheSamePosition

`place` reuses the slot at the position once; the second call for the same position creates one.

### AChangedShapeGetsANewSlotAndReleasesTheOld, AChangedElementTypeGetsANewSlot

A slot is reused only for the same extents and element type. The slot left behind is returned by
`end_step`.

### ASlotIsClaimedOnce

Two buffers that the same reader slot names get two slots, and the first is not handed out again
through `keep`.

### ASlotOfAnotherProducerIsNotTaken

Another region now feeds the reader's first import. It gets a new slot, and the first producer
keeps its own.

### AnUnclaimedSlotIsReleasedAtTheEndOfTheStep

A step that asks for nothing releases every slot of the step before.

### ForgetReturnsEverySlotOnce

The second step is left open after one `keep` and one creation, as a step that failed after
planning leaves it. `forget` returns the kept slot and the created one, each once.

## test_storage_bank.cpp

Suites: `Unit_StorageBank`.

Tests that `StorageBank` holds, allocates and lets go of tensors by storage id.

This file needs an engine, because the bank allocates tensors.

### AllocateHoldsATensorOfTheShapeAndTypeAsked

The tensor has 12 elements of FLOAT64, and `allocation_count()` is 1.

### AdoptSharesTheStorageOfTheCallersTensor

The bank's tensor has the address of the caller's and reads the caller's numbers. `adopt` does
not count as an allocation.

### ReleaseForgetsTheTensorAndIgnoresAnUnknownId

`release` of an id the bank does not hold does nothing, which lets a session release the ids a
failed step never allocated.

### ASecondTensorUnderOneIdStopsTheProcess, AskingForAnUnknownIdStopsTheProcess

Both are caller errors and are fatal.

## test_session.cpp

Suites: `Unit_EagerSession`.

Tests the steps a `Session` runs and the points at which a `Step` runs them.

This file needs an engine. Its cases are the acceptance list of
`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`, section 10, as far as one module reaches.

### eager_run_support.hpp

`Support::Loop` owns a `ScopePartitioner`, a `RegionExecutor`, a `Session`, a model of linear
layers and one batch. `step(options)` traces the model, differentiates, applies the update and
reads the loss. `TraceOptions::scoping` says where scopes open: nowhere, one per layer, one
between the first layer and its activation, or "encoder" and "head". `PlainDescent` is the
update: one `axpy` of minus the rate times the gradient per parameter. It is a
`Graph::StepRecorder` and counts its recordings and its replays. `make_model` fills every tensor
with `fill_wave`, so two models of the same widths start from the same numbers.

### Compiled

`make_compiled` traces the same two-layer model in the same order, differentiates it, runs the
cast and fusion passes with the leaves, the loss and the gradients observed, plans the memory
with the loss and the gradients pinned, binds the same tensors and compiles the graph with a
`PlainDescent` behind it through `Graph::CompiledStep`.

### AnUnscopedStepGivesTheLossBytesOfTheCompiledStep

Four steps of both paths give the same loss bytes. With no scope the eager step is one region
that holds the forward and the backward nodes, so the two paths lower the same nodes. The case
compares with `Graph::CompiledStep` and not with `Train::Trainer`, which a test of this module
cannot name.

### ASettledLoopComposesOnceAndAllocatesNothingMore

A three-layer model with one scope per layer runs five steps. After the first, the composition
count stays 1 and the lowering count, the pool bytes, the session's allocation count and its
tensor count do not move. The update is recorded once and replayed five times.

### FlippingABranchRebuildsThatScopesTwoRegionsAndNoOther

Two steps with a K_TANH head, then one with a K_RELU head. The head's forward and backward
regions are lowered twice and every other region once. The loss is traced after the head closes,
in the second unnamed segment of the step, so its region is named "#1"; the first unnamed
segment holds the input leaf and no region.

### AScopeBetweenALinearLayerAndItsActivationTrainsToTheSameLoss

Two loops start from the same numbers, one with no scope and one with a scope between the first
layer and its activation. Their losses are compared within `K_LOSS_TOLERANCE`, 1e-5, over five
steps, and not byte for byte: the two loops run different kernels for the first layer.
No region of the scoped loop reports a `GEMMBiasReLU`.

### AScopeThatAppearsAndDisappearsRecordsTheRegionsItRenumbers

A scope "extra" appears in the second step and is gone in the third. The trailing key is twice
the segment count, so the update is recorded three times. The encoder's forward region keeps its
key and is lowered once; its backward region moves twice and is lowered three times.

### AParameterNoStepTracesKeepsAZeroGradientAndItsValue

A third layer is bound and paired with the update and never traced. After two steps its gradient
tensor holds zeros and its weight holds the bytes it started with.

### AStepWithoutBackwardRunsItsForwardNodesWhenAValueIsRead

`read_scalar` of an untouched step runs it. No update was applied, so none was recorded.

### AValueTheStepDidNotKeepCannotBeReadAfterTheRun

After `apply` the model's result is not readable: it sits inside the step's one region and was
not passed to `keep`. The loss is, because `backward` keeps it. `keep` after the run is refused.

### TracingAfterTheStepRanIsRefused

A node emitted after `finish` makes the next read return `Unsupported`.

### AStateTheSessionDoesNotHoldIsRefused, BindingAStateTwiceIsRefused

`Step::state` and `Session::gradient` refuse an identity that was never bound, and
`Session::bind_state` refuses one that already is.

### ASecondLiveStepStopsTheProcess

A session traces one step at a time.

Suites: `Unit_EagerStepCost`.

Prints the host cost of an eager step and asserts only that the measured steps were the kind it
claims to measure.

This file needs an engine. Each row it prints starts with `[eager step cost]` and gives the mean
over the measured steps, in microseconds, of the three host stages `StepCost` separates
(partition, plan, compile), their sum, and the whole step beside them. The numbers are read off
a run; none is asserted and none is recorded in this page.

`K_WARM_STEPS`, 3, run before the measured ones so every region is recorded. `K_MEASURED_STEPS`
is 20. `K_RATE` is 1e-6 so that fifty RELU layers stay finite for the length of the run.
`widths_of(layers)` gives a model 16 wide with 4 results.

### AnUnchangedStepOfTheTwoLayerModel, AnUnchangedStepOfAFiftyLayerModel

`measure_settled` runs a loop with no scope and with one scope per layer. It returns
`Incompatible` when a measured step built or recorded a region, since that step would not be an
unchanged one.

### AScopeThatAppearsAndDisappears

A fifty-layer loop with one scope per layer alternates a step with a scope "extra" and a step
without it, ten times each, and prints the two means. The case fails when no region was recorded
again, because the steps would then not have been renumberings.
