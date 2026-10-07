# eager

The host-lane half of the eager front end: tracing scopes, cutting a differentiated graph into
regions, and comparing a region with the one a previous step produced. Nothing here reaches an
engine or SYCL, so every unit is tested without a device.

The module depends on `graph`, `autograd` and `core`. Nothing depends on it yet. The design is
`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`; the units that execute a partition are not built.

The sections below record why the code is the way it is. Each is named after a header and the
symbol it explains.

## segment.hpp

### Segment

A segment is a maximal run of consecutively traced nodes that share one innermost open scope.
Opening or closing a scope ends the running segment, and a run with no node is dropped. Segments
are therefore contiguous, ordered by node id, and together cover every traced node exactly once.
That is what lets a partitioner number them without inspecting a single dependency: a node can only
read values traced before it.

The name is the path of open scopes joined by `/`, and it is empty outside any scope. `occurrence`
counts earlier segments of the same name in the same trace, so a scope opened in a loop, or an
outer scope that resumes after an inner one closes, yields distinct segments with one name.

## scope_trace.hpp

### ScopeTrace

`ScopeTrace` does not observe node emission. It reads the graph's node count when a scope opens,
when one closes and at `seal()`, and that is enough to place every boundary. The builder and the
layers therefore need no change to be traced under scopes.

A scope name may not be empty and may not contain `/` or `#`, because the first joins a path and
the second separates a name from its occurrence in a region name.

## scope.hpp

### Scope

`Scope` exists so that a scope cannot be left open by an early return or an exception between
`open()` and `close()`.

## autograd/provenance.hpp

### Provenance

The partitioner must put a backward node in the backward region of the segment whose forward node
it differentiates. Operands do not say which that is. The weight gradient of a linear layer reads
the layer's input, which an earlier segment produced, so a rule based on operands would file it
under the wrong segment. The ordering would still be valid; the locality would be lost, and a
change in one scope would rebuild its neighbour.

Only the reverse walk knows which forward node a rule is running for, so `differentiate()` records
it. Every node appended while the walk is at forward node `f` has origin `f`: the nodes `f`'s rule
emits, and the `ACCUMULATE` nodes emitted when the gradients of `f`'s outputs are summed. An
`ACCUMULATE` emitted by the closing read of a requested gradient has the producer of that value as
its origin, which for a parameter is its `LEAF` node.

## partition.hpp

### RegionKey

SushiRuntime orders two regions that share bytes by ascending key: the lower key is the producer.
A partition is valid when no node reads a value that a higher key produces.

### Partition::optimizer_key

The parameter update reads every gradient and writes every parameter, so it needs a key above
every region. The partition reserves one and records nothing under it.

## scope_partitioner.hpp

### ScopePartitioner

With `m` segments in trace order, a forward node of segment `k` (counted from 1) gets key `k`, and
a backward node whose origin lies in segment `k` gets key `2m - k`. Later segments get lower
backward keys, which matches the direction gradients flow: the backward region of segment `k` reads
gradients that the backward region of segment `k + 1` produced.

For `k = m` the two keys coincide, so the last segment's forward and backward nodes share a region.
When the caller scopes the loss, that region is the loss and the seed of its gradient.

A backward node whose origin is a `LEAF` sums contributions from several segments. It takes the
highest key among the regions of its operands' producers, so it runs after all of them.

`LEAF` nodes get `K_NO_REGION`. The lowering records nothing for a leaf.

The optimizer key is `2m`.

## region_signature.hpp

### RegionSignature

Two steps produce the same region when their signatures compare equal. A match is trusted, and a
recorded kernel is replayed without being looked at again, so a false match would run the wrong
kernels with no error. The signature therefore carries the operation list beside the hash, and
equality compares both.

### signature_of

The hash covers, for the region's nodes in order: the operation, the attributes, and for each
result its shape, element type and layout; then each operand's position.

An operand's position is never its value id or its producer's node id. Those shift whenever an
earlier region grows, and a region whose own structure did not change must keep its signature.

An operand produced inside the region is hashed by the index of its producer among the region's
nodes and its index among that producer's results.

Every other operand is external to the region: a leaf, or a value another region produces. It is
hashed by the identity the caller supplies for it, and by nothing else. A recorded kernel holds the
address of each operand, so what a region needs to know about an external operand is whether it is
still the same storage, not where in the graph it came from. The caller owns that storage and is
the only party that can say. A parameter keeps its storage for the run. A value that crosses a
region boundary keeps the storage the session bound it to for as long as the session keeps it, and
that is what lets a region survive the rebuild of the region that feeds it.

The function knows nothing about other regions for the same reason. Hashing a boundary operand by
its producer's region and the producer's index inside it does not work: the index moves whenever
the producer region grows before its last node, so every reader of a changed region would change
with it.
