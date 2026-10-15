# tests/core/graph/test_ir.cpp


Suites: `Unit_Ir`, `Unit_IR`.

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

## AddValueRecordsShapeDTypeAndLifetime

`add_value` of a [2, 3] `Float64` parameter named "w" gives a graph of one value with that
shape, element type, lifetime and debug name and no producer.

## AddNodeClaimsItsOutputs

After `add_node` of an ADD over two inputs, the output value's producer is that node and the
node has two inputs.

## ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

## ReferencingAnUnknownValueThrows

`add_node` throws `Error` when it reads the value 99 and when it writes the value 99.

## PropagateRequiresGradFlowsForwardThroughTheWholeChain

Before `finalise`, the MATMUL of an input and a parameter and the ReLU after it do not
require a gradient. After it the input still does not, and the parameter, the product and the
ReLU do.

## PropagateRequiresGradLeavesAFrozenBranchAlone

After `finalise`, a ReLU of the input does not require a gradient and the MATMUL of the input
and a parameter does.

## BuilderInfersMatmulShapesForEveryTranspositionCase

With [4, 3] or [3, 4] on the left and [3, 5] or [5, 3] on the right, `matmul` gives [4, 5]
for each of the four combinations of the two transpose flags.

## BuilderRejectsAContractionMismatch

`matmul` of [4, 3] and [7, 5] throws `Error`.

## BuilderChecksBiasAndLossShapes

Over a [4, 5] value, `bias_add` with a [5] bias gives [4, 5] and with a [4] bias throws
`Error`. `sum_rows` gives [5]. `nll_loss` with [4, 5] targets gives [1] and with [4, 6]
targets throws `Error`.

## BuilderRejectsMixedElementTypes

`add` and `matmul` of a `Float32` and a `Float64` operand both throw `Error`.

## NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

## InsertNodeRejectsAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

`node_count()` is still 1 after the throw. The insert was rejected, not half-applied: nothing
was inserted and no producer was claimed.

The final `EXPECT_NO_THROW` inserts the same node ahead of that reader and succeeds. That makes
the rejection a statement about position and not about the node.

## InsertNodeRejectsAnOperandProducedNoEarlierThanItself

With one RELU node writing value 1, `insert_node` at 0 of a TANH that reads value 1 raises an
error holding "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes" and "topological".
The graph still has one node and the TANH's output has no producer.

## AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::K_CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
the lowering reads each input through `LoweringContext::input` and passes the tensor straight to
SushiBLAS.

## ABufferLeafIsPersistentAndNeedsNoGradient

The nine cases from here are in the suite `Unit_Ir`; the cases above are in `Unit_IR`.

`GraphBuilder::buffer` declares a value of the shape, type and name it was given, with the
lifetime `Lifetime::Buffer` and `requires_grad` false. Its producer is a K_LEAF node. A K_BIAS_ADD
that reads an input and a buffer does not require a gradient either. `is_persistent(BUFFER)` is
true and `lifetime_name(BUFFER)` is "buffer".

## ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist

`Graph::is_persistent` is true for PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT and BUFFER, and
false for ACTIVATION and SCRATCH.

## StateLeafReturnsTheBoundLeafAndNoValueForAnUnboundState

`Graph::state_leaf` returns `K_NO_VALUE` before any binding. After `bind_state_leaf` of state 7
to a buffer and state 5 to a parameter, it returns each leaf for its state and `K_NO_VALUE` for
state 6 and for `K_NO_STATE`. `state_leaves()` holds the two pairs in the order they were
bound, not in the order of the state numbers.

## BindStateLeafRefusesASecondBindingOfAStateOrOfALeaf

With state 7 bound to a leaf, four bindings throw: state 7 to another leaf ("State 7",
"already has leaf"), state 8 to the bound leaf ("already stands for state 7"), `K_NO_STATE` to
a leaf ("no identity"), and state 9 to value 99 ("value 99", "does not exist"). The table
still holds the first pair alone.

## WithoutNodesKeepsEveryValueAndStateLeafAndHoldsNoNode

`Graph::without_nodes()` returns a graph with no node and the same number of values. Every
value keeps its shape, element type, layout, lifetime, `requires_grad` and debug name, and its
producer is `K_NO_NODE`. The fixture holds an NCHW value, a parameter that requires a gradient
and a HALF buffer, and each is checked by name. The copy's `state_leaves()` equals the
source's. The source graph still has its nodes and producers.

## ARefusalIsReportedAfterItIsMarkedAndSurvivesWithoutNodes

`has_refusal()` is false for a new graph and for its `without_nodes()` copy. After
`mark_refused()` it is true for both.

## StateIdentityDrawsOneIdentityPerHolder

`Graph::StateIdentity` in `modules/core/graph/include/SushiAI/graph/state_id.hpp` holds the identity of one piece
of model state. `peek()` returns `K_NO_STATE` before the first `get()`. `get()` returns the
same identity on every call, `peek()` then returns it too, and a second holder's `get()`
returns a different one.

## ACopiedStateIdentityDoesNotShareTheSourcesIdentity

A holder copy-constructed from one that has drawn peeks `K_NO_STATE` and then draws a
different identity. A holder that had drawn and is then copy-assigned from the source also
peeks `K_NO_STATE`, and its next identity differs from the source's and from the one it had.
The source keeps its identity.

## AMovedStateIdentityCarriesItsIdentityAway

After a move construction the target peeks the source's identity and the source peeks
`K_NO_STATE`. A move assignment over a holder that had drawn does the same.
