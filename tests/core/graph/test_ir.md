# tests/core/graph/test_ir.cpp


Suites: `Small_Ir`, `Small_IrMatmulRow`, `Small_IrLifetimeRow`, `Medium_IrFatal`.

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

## AddValueRecordsShapeDTypeAndLifetime

`add_value` of a [2, 3] `Float64` parameter named "w" gives a graph of one value with that
shape, element type, lifetime and debug name and no producer.

## AddNodeClaimsItsOutputs

After `add_node` of an ADD over two inputs, the output value's producer is that node and the
node has two inputs.

## ASecondProducerForOneValueIsFatal

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write stops the process and is not tolerated.

## ReferencingAnUnknownValueIsFatal

`add_node` stops the process when it reads the value 99 and when it writes the value 99.

## PropagateRequiresGradFlowsForwardThroughTheWholeChain

Before `finalise`, the MATMUL of an input and a parameter and the ReLU after it do not
require a gradient. After it the input still does not, and the parameter, the product and the
ReLU do.

## PropagateRequiresGradLeavesAFrozenBranchAlone

After `finalise`, a ReLU of the input does not require a gradient and the MATMUL of the input
and a parameter does.

## BuilderInfersMatmulShapesForEveryTranspositionCase

A parameterised case with one row per combination of the two transpose flags, named
`NoTranspose`, `TransposedLeft`, `TransposedRight` and `TransposedBoth`. Each row feeds
[4, 3] or [3, 4] on the left and [3, 5] or [5, 3] on the right, and `matmul` gives [4, 5].

## BuilderRejectsAContractionMismatch

`matmul` of [4, 3] and [7, 5] fails with `InvalidDescription`.

## BiasAddKeepsTheInputShape

`bias_add` of a [4, 5] value and a [5] bias gives [4, 5].

## BiasAddRefusesABiasOfTheWrongExtent

`bias_add` of a [4, 5] value and a [4] bias fails with `InvalidDescription`.

## SumRowsYieldsOneElementPerColumn

`sum_rows` of a [4, 5] value gives [5].

## NllLossYieldsOneElement

`nll_loss` of [4, 5] logits and [4, 5] targets gives [1].

## NllLossRefusesTargetsOfAnotherShape

`nll_loss` of [4, 5] logits and [4, 6] targets fails with `InvalidDescription`.

## BuilderRejectsMixedElementTypes

`add` and `matmul` of a `Float32` and a `Float64` operand both fail with `InvalidDescription`.

## NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

## InsertNodeIsFatalForAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

## InsertNodeAcceptsAResultAheadOfItsReader

The same node inserted at 0, ahead of the reader, succeeds: the graph has two nodes, the result's
producer is node 0 and the reader's output has node 1 as producer. That makes the refusal above a
statement about position and not about the node.

## InsertNodeIsFatalForAnOperandProducedNoEarlierThanItself

With one RELU node writing value 1, `insert_node` at 0 of a TANH that reads value 1 raises an
report holding "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes" and "topological".

## AnAttributeReadPastTheLastSlotYieldsZero

`Attributes::get` of `K_CAPACITY`, one past the last slot, returns 0 while every slot holds 1.
Out-of-range reads stay defined.

`Attributes::K_CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
the lowering reads each input through `LoweringContext::input` and passes the tensor straight to
SushiBLAS.

## ABufferLeafIsPersistent

Every case of this file is in the suite `Small_Ir`, except the three cases that stop the process
(`ReferencingAnUnknownValueIsFatal` and the two `BindStateLeafIsFatal...` cases), which are in
`Medium_IrFatal`. Each death check runs in a child process, so these cases exceeded the 0.1 s
limit of a `small` test on the debug build (RW-166); a death test is `medium` by its size rule.

`GraphBuilder::buffer` declares a value of the shape, type and name it was given, with the
lifetime `Lifetime::Buffer`. Its producer is a K_LEAF node. `is_persistent(BUFFER)` is true and
`lifetime_name(BUFFER)` is "buffer".

## ABufferLeafNeedsNoGradient

A buffer has `requires_grad` false after `finalise`, and a K_BIAS_ADD that reads an input and a
buffer does not require a gradient either.

## ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist

A parameterised case with one row per lifetime, named after it. `Graph::is_persistent` is true
for PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT and BUFFER, and false for ACTIVATION and SCRATCH.

## StateLeafReturnsTheBoundLeaf

After `bind_state_leaf` of state 7 to a buffer and state 5 to a parameter, `Graph::state_leaf`
returns each leaf for its state.

## StateLeafReturnsNoValueForAnUnboundState

With state 7 bound, `state_leaf` returns `K_NO_VALUE` for state 6 and for `K_NO_STATE`.

## StateLeavesListTheBindingsInBindingOrder

`state_leaves()` is empty before any binding. After state 7 and then state 5 are bound it holds
the two pairs in the order they were bound, not in the order of the state numbers.

## BindStateLeafIsFatalForASecondBindingOfAStateOrOfALeaf

With state 7 bound to a leaf, two bindings stop the process: state 7 to another leaf ("State 7",
"already has leaf") and state 8 to the bound leaf ("already stands for state 7").

## BindStateLeafIsFatalForNoIdentityOrAnUnknownValue

`K_NO_STATE` bound to a leaf stops the process ("no identity"), and so does state 9 bound to value
99 ("value 99", "does not exist").

## WithoutNodesHoldsNoNode

`Graph::without_nodes()` returns a graph with no node and the same number of values. Values of
the copy have `K_NO_NODE` as producer. The fixture holds an NCHW value, a parameter that
requires a gradient and a HALF buffer, with two state leaves bound.

## WithoutNodesKeepsEveryValueField

Each fixture value keeps its shape, element type, layout, lifetime, `requires_grad` and debug
name in the copy, and the NCHW layout, the gradient flag and the HALF type are checked by name.

## WithoutNodesKeepsTheStateLeaves

The copy's `state_leaves()` equals the source's, and `state_leaf` of state 12 returns the buffer.

## WithoutNodesLeavesTheSourceGraphUnchanged

The source graph still has its nodes and producers after the copy.

## ARefusalIsReportedAfterItIsMarked

`has_refusal()` is false for a new graph and true after `mark_refused()`.

## AMarkedRefusalSurvivesWithoutNodes

`without_nodes().has_refusal()` is false before `mark_refused()` and true after it.
