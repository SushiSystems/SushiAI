# tests/core/graph/lowering/test_context.cpp


Suites: `Small_LoweringContext`, `Medium_LoweringContext`.

Tests what `LoweringContext` gives a lowering row. The fixture traces one RELU over a [2, 3]
input, materialises its leaves in a `TensorPool` and binds a context to the RELU node with the
engine's recorder and random stream. The file-local `FixedSelector` answers every query with one
form. No case records a task.

## TheTransposeFlagMapsOntoSushiBLAS

`transpose_of` maps true to `Transpose::Trans` and false to `Transpose::NoTrans`, checked at
compile time.

## TheContextDescribesItsNode

The context returns the node id it was bound to, the graph, the RELU operation and its declared
name, the [2, 3] operand and result, and the pool's tensor for the operand.

## AResultIsAllocatedOnTheFirstRequest

The result is unbound before `output(0)`, bound in the context and in the pool after it, and a
second request returns the same tensor.

## AnAliasedResultSharesItsOperandsStorage

After `alias_output(0, 0)` the result is bound and its tensor has the operand's storage.

## CountAddsToThePassTotal

With the pass counter at 2, `count(3)` makes `emitted()` and the counter itself 5.

## ASelectionIsRecordedWithTheTasksItCost

`select` of a GEMM, bias and ReLU match returns that form and appends one `FusionSelection`
naming it as matched and chosen, with a reason. After four counted tasks, `close_selection` with
the count taken before sets the selection's task count to 4.

## ASelectorMayNarrowAGEMMEpilogueToTheBiasForm

A selector that answers the plain GEMM and bias form to a GEMM, bias and tanh match is accepted,
and the selection records the tanh form as matched and the bias form as chosen.

## APreludeNeedsAListToGoInto

A context bound without a list of preludes refuses `before_each_replay` with `Unsupported` and a
message naming `CompiledStep`. A context bound with a list appends the prelude, and calling it
with 7 reaches the callable.
