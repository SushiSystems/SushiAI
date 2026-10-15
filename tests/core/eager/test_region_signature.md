# tests/core/eager/test_region_signature.cpp


Suites: `Unit_RegionSignature`.

Tests the signature that `signature_of` computes for one region of a partition.

This file needs no device. `Eager::signature_of` reads a graph, a partition and one identity per
value, all of which a case builds on the host. `modules/core/eager/README.md`, under
`region_signature.hpp`, lists what the hash covers and gives the reason for each operand rule.

Most cases build two graphs that differ in one property and compare the two signatures. Where
the operation list is the same in both, the case asserts that first, so the difference it then
asserts is a difference in the hash alone.

## Region keys and identities

`K_BODY` is key 1, the one region of a hand-built partition, and `K_READER` is key 2, the region
that reads from it in a two-region partition. `K_PARAMETER_IDENTITY` and `K_EXTERNAL_IDENTITY`
are the default identities of the parameter and of the value `K_READER` reads from outside.
They are stated once so that a case names the one it changes.

## Traced

`Traced` holds a graph, its partition and one identity per value id, which are the three things
`Eager::signature_of` reads.

## signature_of

The file-local `signature_of` has three overloads. The first takes a `Traced` and a region key
and calls `Eager::signature_of` with the identity vector as a span. The other two take a
`Recipe` or an `ExternalRecipe`, build its graph and return the signature of `K_BODY` or of
`K_READER`.

## Recipe

`Recipe` describes `scale(combine(relu(x), tanh(x)), alpha)` over a parameter `x`, and each of
its fields is one property a case varies: the shape, the element type, the K_SCALE factor, K_SUB in
place of K_ADD, the operand order of the combine, and the identity of `x`. `build(Recipe)` puts
the four operations under `K_BODY` and the K_LEAF under no region, and gives every value but `x`
identity zero.

## node_of

`node_of` returns the first node of a graph that runs a given operation. The attribute and
layout cases use it to reach the node they edit after `build` returns.

## assign_regions

`assign_regions` gives a `Traced` its partition from one key per non-K_LEAF node, in node order,
and a region list. A K_LEAF gets `K_NO_REGION`. It also resets every identity to zero, so a
builder that calls it sets afterwards only the identities its case is about.

## ExternalRecipe

`ExternalRecipe` describes a graph whose `K_READER` region holds one K_SCALE that reads a value
the region does not produce. Its fields choose whether that value is the parameter `x` or the
K_RELU result of `K_BODY`, whether `K_BODY` appears in the region list, and which identity the
value has.

## ProbeOperands

`ProbeOperands` names four operand orders for a hand-built three-operand K_ADD in `K_READER`.
`build(ProbeOperands)` leaves every identity at zero, and the one in-region operand is result 0
of the region's node 0. Both orders of a pair therefore feed the hash the same words, and the
two signatures can differ only if the kind of each operand enters the hash.

## build_reader_of_result

`build_reader_of_result` adds a K_RELU node by hand with two results and a K_TANH that reads the
result at the given index, both under `K_BODY`. The K_RELU is added through `Graph::add_node`
because `GraphBuilder::relu` returns one result.

## with_identity

`with_identity` records an identity for one value and returns the value id, so that
`ScopedPerceptron` can assign an identity in the expression that declares a leaf.

## info_of

`info_of` returns the `RegionInfo` of a key, or null when the partition does not list the key.

## role_identity

`role_identity` hashes a region's name, its direction, how many nodes of the region follow the
producing node, and the index of the value among that node's results. Counting from the
region's end keeps the identity of a region's last result the same when nodes are added before
it.

## identify_produced_values

`identify_produced_values` gives every value that a region's node writes its `role_identity`.
In the tests it stands in for the caller of `signature_of`, which owns the storage a boundary
value is bound to; `modules/core/eager/README.md` says why only the caller can supply that
identity.

## ScopedPerceptron

`ScopedPerceptron` traces three scopes, "stem", "body" and "head", then differentiates and
partitions the graph with `ScopePartitioner`. Its argument is the number of linear layers in
"stem". Leaves get fixed identities and produced values get role identities. `key_of` returns
the key of a region by name and direction, and `stem_output` is the value "body" reads from
"stem".

## TheSameTraceGivesTheSameSignature

Two graphs built from the default `Recipe` have equal hashes and compare equal. The operation
list is K_RELU, K_TANH, K_ADD, K_SCALE, which is the region's nodes in order with the K_LEAF left out.

## AChangedAttributeChangesTheHash

A K_SCALE factor of 3.0 in place of 2.0 leaves the operation list equal and makes the hash
differ.

## AChangedLastAttributeSlotChangesTheHash

The case writes 1.0 into slot `Attributes::K_CAPACITY - 1` of the K_SCALE node and expects another
hash. It covers the last slot because a hash that read a fixed number of leading slots would
agree with the other attribute case and miss this one.

## AChangedLayoutChangesTheHash

The K_RELU result is retagged NCHW in one of two otherwise equal graphs, and the hashes differ.
Shape and element type are the same in both, so the layout is the only input that moved.

## AChangedShapeChangesTheHash

A shape of [4, 5] in place of [4, 3] leaves the operation list equal and makes the hash differ.

## AChangedDTypeChangesTheHash

FLOAT64 in place of FLOAT32 leaves the operation list equal and makes the hash differ.

## AChangedOpChangesTheHash

K_SUB in place of K_ADD makes both the operation list and the hash differ. It is the one
`Recipe` case in which the two lists are unequal.

## ADifferentParameterChangesTheHash

The same structure over a parameter with identity `K_PARAMETER_IDENTITY + 1` has the same
operation list and another hash. The parameter is external to the region, so its identity is
the only thing the hash holds about it.

## AnEarlierRegionGrowingDoesNotChangeALaterRegion

`ScopedPerceptron(1)` and `ScopedPerceptron(2)` differ by one linear layer in "stem". The three
`ASSERT` lines at the top establish that the case tests what it claims: the value "body" reads
has a different id in the two graphs, the same identity in both, and the forward and backward
regions of "body" have different keys.

For each direction, the signature of "body#0" has a non-empty operation list and is equal in
the two graphs, by hash and by `operator==`. The forward signature of "stem#0" differs between
them, which shows that the two graphs are not equal throughout.

## ARewiredOperandChangesTheHash

`add(relu(x), tanh(x))` and `add(tanh(x), relu(x))` have the same operation list and different
hashes. Both operands are produced inside the region, so the case covers the in-region operand
position.

## AReadOfAnotherResultChangesTheHash

A K_TANH that reads result 1 of a two-result node hashes apart from one that reads result 0. The
producer's index in the region is the same in both, so the result index is what separates them.

## ABoundaryValueOfAnotherIdentityChangesTheHash

The K_SCALE in `K_READER` reads the K_RELU result of `K_BODY`. With identity
`K_EXTERNAL_IDENTITY + 1` for that value, the operation list is equal and the hash differs.

## ALeafAndABoundaryValueOfOneIdentityHashAlike

The reader's signature is equal, by hash and by `operator==`, whether its operand is the
parameter `x` or the result of another region, as long as both carry `K_EXTERNAL_IDENTITY`.
Where an external operand comes from in the graph does not enter the hash.

## AnOperandFromAnUnlistedRegionHashesAsExternal

The same graph is signed with `K_BODY` present in the region list and with it absent, and the
two signatures of `K_READER` are equal. An operand whose producer has a key other than the
region's is external whether or not the partition lists that key.

## AnOperandInsideTheRegionHashesApartFromOneOutside

The probe node reads one result of `K_READER` and the result of `K_BODY` twice, in the two
orders `InsideThenOutside` and `OutsideThenInside`. The operation lists are equal and the
hashes differ.

## ALeafOperandHashesApartFromOneInsideTheRegion

The same pair of orders with the leaf `x` in place of the `K_BODY` result,
`InsideThenLeaves` and `LeavesThenInside`, also gives equal operation lists and different
hashes.

## EqualHashesWithDifferentOpsCompareUnequal

Three signatures are written by hand with hash 42. The one listing K_ADD and the one listing K_SUB
compare unequal under `==` and `!=`, and two that both list K_ADD compare equal. A real collision
cannot be produced on demand, so the case constructs one.

## ARegionThePartitionDoesNotListThrows

`signature_of` throws for key 6 and for `K_NO_REGION`. Each message names the key, "Region 6"
and "Region 0", and contains "not listed".

## APartitionOfAnotherNodeCountThrows

One entry is removed from `region_of_node`, which leaves four entries for a five-node graph.
The message contains "assigns 4 nodes" and "holds 5".

## AnExternalValueWithoutAnIdentityThrows

The identity vector is cleared, so the id of `x` lies beyond its end. The message contains
"External value 0" and "no identity". The call throws and does not read past the span.
