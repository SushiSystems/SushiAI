# tests/core/graph

The tests of the module `core/graph`, one file per header of `modules/core/graph/include/`. The suite prefix of a case gives its ctest label.

## test_compiled_step.cpp

Suites: `Unit_CompiledStep`.

Tests `CompiledStep::restore_run_count` and the node range of `CompiledStep::compile` in
`modules/core/graph/include/SushiAI/graph/compiled_step.hpp`. The fixture is an `EngineTest`.
Each case compiles one dropout of rate 0.25 over a leaf of 256 ones with a fixed site seed, so
the step's output is the mask: 4/3 where `dropout_keeps` keeps an element and 0 elsewhere,
compared within 1e-6.

### ARestoredRunCountIsTheIndexOfTheNextRun

One step runs three times. A second step, with a run count of 0, is restored to 2 and runs
once. Its run count is then 3, its output equals the first step's third output, and that
output is the mask `dropout_keeps` gives for run 2.

### ARestoreAfterARunIsRefusedWithTheCount

After one run, `restore_run_count(4)` fails with `InvalidDescription`, "before its first run"
and "run 1", and the run count stays 1.

### ASecondRestoreIsRefused

`restore_run_count(0)` and then `restore_run_count(5)` succeed. A third call with 7 fails with
`InvalidDescription` and "run 5", and the run count stays 5.

### TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst

A graph scales a leaf of four elements by 2 and the result by 3. The nodes up to the producer
of the doubled value and the nodes behind it compile into two steps over one pool, each with
at least one task. After the first step the doubled value holds 2, 4, 6, 8. The host then writes 10 into every element of it, and
after the second step the tripled value holds 30 four times.

### ARangeOutsideTheGraphIsRefused

A range that ends one node behind the last fails with `InvalidDescription` and "does not lie
in a graph", and the range [1, 0) with `InvalidDescription` and "[1, 0)".

## test_dropout_mask.cpp

Suites: `Unit_DropoutMask`.

Tests the functions of `modules/core/graph/include/SushiAI/graph/dropout_mask.hpp`:
`splitmix64_at`, `dropout_site_seed`, `dropout_bits`, `dropout_keeps` and `fill_keep_mask`.
Every case but the first two draws from the site seed 2026. A mask is compared as a string of
'1' for a kept element and '0' for a dropped one.

### TheSequenceOfSeedZeroIsThePublishedSplitMix64

`splitmix64_at(0, i)` for i from 0 to 3 is 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
0x06C45D188009454F and 0xF88BB8A8724C81EC.

### EachSiteOfATraceHasItsOwnSeed

`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are the first two numbers of the case
above. `dropout_site_seed(7, 0)` is 0x63CBE1E459320DD7 and `dropout_site_seed(7, 1)` is
0x044C3CD7F43C661C.

### TheBitsOfASeedARunAndAnElementAreGolden

`dropout_bits` for elements 0 to 3 of runs 0 and 1 equals eight 64-bit constants written in the
test.

### TheMaskOfARunIsGolden

At a keep of 0.5 the first 32 elements of run 0 are "11110000011001111110011111110000" and those
of run 1 are "01110100100110000111111111000011".

### TheFillAgreesWithTheElementwiseQuestion

For 257 elements of run 5 at a keep of 0.7, each element of the filled mask is 1 exactly when
`dropout_keeps` returns true for it.

### AShorterFillIsAPrefixOfALongerOne

A fill of 40 elements of run 2 at a keep of 0.9 equals the first 40 elements of a fill of 96.

### TheSameInputsGiveTheSameMask

Two fills of 4096 elements of run 9 at a keep of 0.8 are equal.

### TwoRunsAndTwoSitesDrawDifferentMasks

Runs 0 and 1 at a keep of 0.5 differ in 2033 of 4096 elements. The masks of the site seeds
`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are not equal.

### TheKeptFractionIsTheKeepProbability

Over 2^20 draws the kept fraction is within 0.002 of the keep: 0.9 and 0.5 in run 0, and 0.9 in
run 3.

### TheTwoEndsOfTheKeepRangeAreExact

A keep of 1 keeps all 64 elements and a keep of 0 keeps none.

## test_fusion.cpp

Suites: `Unit_Fusion`.

Tests the matches and the refusals of the static fusion pass on synthetic graphs.

This file needs no device. The static fusion pass is a pure function of the IR, as the memory
planner is, so every case is a synthetic graph and a table of expectations with no engine
anywhere.

The cases pin the set of refusals more than the matches. A pass that fuses too little is slow.
A pass that fuses too much is silently wrong: the graph still runs, the loss still falls, and
the only symptom is a model that converges somewhere slightly different. So most cases are
graphs the pass must decline, one per rule:

- an intermediate with a second consumer (fusing it means losing it),
- an intermediate the caller reads by id (the same loss, seen from outside),
- a spilled pre-activation read by a node that runs before the fused one.

The activation whose gradient differentiates at its input used to be a fourth entry on that
list. It no longer is. K_GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal became a condition: an INPUT-gradient activation may be folded only by an
entry that writes the tensor its gradient reads, and only if it writes the right one. That
condition has a compile-time half in `modules/core/graph/source/fusion.cpp`, where an entry breaking either half
fails a static_assert. A test cannot cover that half, because a test that fails to compile is
not a test. This file checks the pair of properties the assertion reads and, above all, that the
fold leaves the pre-activation written and readable by the backward node.

The cases fall into four groups, in file order:

- the catalog is well formed
- the matches
- the refusals, which are the half that matters
- the pass keeps the graph usable

### EveryFormRoundTripsThroughItsOperationIdentity

The lowering dispatches on `fused_form_op` and `fused_form_of`. A form that does not round-trip
would lower as "no recorder for this operation", which is loud, but only at run time and only on
a graph that happens to contain it.

### AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts

`activation_survives_fusion(K_RELU)` is the property the static_assert in `modules/core/graph/source/fusion.cpp`
reads. If it ever flips for an activation that has a catalog entry, the static_assert catches
it, but only while the property stays true of the table the static_assert consults. This test
pins that.

`activation_survives_fusion(GELU)` is false because SushiBLAS's `gelu_backward` takes the
pre-activation. An entry that wrote only its result would differentiate at a tensor that never
reached memory. That has not changed and must not, because it is the premise the static_assert
reasons from.

`catalog_folds_activation(GELU)` is true because the conclusion drawn from that premise
changed. The one entry that spills folds GELU. "Survives fusion" and "the catalog folds it" are
now two questions with two answers, and a caller who reads the first as the second gets the
wrong answer for exactly one activation.

### AMatmulAndABiasAddBecomeOneNode

`report.fused().size()` is 1, and the fused node reads A, B and the bias and still writes the
value the caller named. That is the whole reason ids are preserved.

The `flag(0)` and `flag(1)` checks hold the transposition flags. They have to survive, because
the fused recorder hands them straight to `execute_gemm`, and losing one gives a wrong answer of
exactly the right shape.

### TheLongestMatchWinsOverTheShorterOneInsideIt

Both gemm+bias and gemm+bias+relu match this graph. Rooted at the bias_add, the shorter one
would win and the relu would be left outside for good, which is why the matcher walks roots in
reverse.

### TheCrossEntropyGradientsElementwiseTailBecomesOnePass

With `scale(sub(exp(log_probs), targets), 0.25)` observed, `fuse` reports one
`FusedForm::ExpSubScale`. The node that produces the scaled value reads `log_probs` and
`targets` and has the attribute 0.25, and the two intermediate values have no producer.

### ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

### AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the K_MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

### AnIntermediateTheCallerReadsByIDIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(K_GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
bias_add, whose own intermediate is free.

### AParameterIsProtectedWithoutBeingListed

A persistent value's buffer outlives the step whatever the graph says, so the caller must not
have to remember to list it. In this graph the "intermediate" is a parameter. That is an odd
graph, but it is the shape a caller would get wrong.

The assignment `lifetime = OPTIMIZER_STATE` forces the exp's result into a persistent class, as
a checkpointed activation would be.

### AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills

The graph is the trap the fusion design is arranged around. The differentiation pass has
already emitted `gelu_backward` reading the pre-activation, so that value has two consumers.
Every other entry would stop at that point under rule 1. This entry does not, for the same
reason it is allowed to exist: the fused node keeps writing that value, as a second output,
under the same id.

The check on the fused node's second output is the assertion that carries the test, and a plain
K_GEMM_BIAS_GELU with no spill would fail it. The fused node still writes the pre-activation as
its second output, and it is the same value, so `gelu_backward`'s operand needed no rewriting
and the backward rule needed no change.

`fused()[0].eliminated` holds only the K_MATMUL's result. A report that listed the spilled value
as eliminated would read like the report of the silently wrong version of this pass.

### ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the K_GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the K_GELU's position, so a reader
of the pre-activation sitting between the bias_add and the K_GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(K_GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

### ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The K_GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

### TheGELUFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The K_MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

### DisablingTheGELUFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::Disabled`, is the only one.

### ARecoverableActivationNextToAFusedGEMMIsFoldedNotReported

This case mirrors DisablingTheGELUFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

### DisablingThePassLeavesTheGraphExactlyAsItWas

With `enabled` false over a linear trace followed by a ReLU, the report is not enabled and
lists nothing fused. The graph keeps its node count, its value count and its one MATMUL.

### DisablingOneFormFallsBackToTheShorterOne

With `FusedForm::GEMMBiasReLU` disabled, the report counts no `GEMMBiasReLU` and one
`GEMMBias`, and holds a decline of `FusionDecline::Disabled`.

### ValueIDsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

### TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

### TheReportNamesEverySubtreeItCollapsed

After a default `fuse`, `to_string` holds "gemm+bias+relu" and "dropping value", and
`nodes_after` equals the graph's node count.

### StateLeavesAndTheRefusalMarkSurviveTheRewrite

The graph is a K_MATMUL, a K_BIAS_ADD whose bias is a buffer, and a K_RELU. The weight and the buffer
are bound to the states 21 and 22 and the graph is marked refused before `Graph::fuse` runs.
The report lists one fused form and the graph holds one K_GEMM_BIAS_RELU node, so the pass did
rebuild the graph. Both `state_leaf` lookups and the whole `state_leaves()` table are what they
were. The buffer keeps the lifetime `BUFFER` and no gradient, the weight still requires one,
and `has_refusal()` is still true.

## test_ir.cpp

Suites: `Unit_Ir`, `Unit_IR`.

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### AddValueRecordsShapeDTypeAndLifetime

`add_value` of a [2, 3] `Float64` parameter named "w" gives a graph of one value with that
shape, element type, lifetime and debug name and no producer.

### AddNodeClaimsItsOutputs

After `add_node` of an ADD over two inputs, the output value's producer is that node and the
node has two inputs.

### ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

### ReferencingAnUnknownValueThrows

`add_node` throws `Error` when it reads the value 99 and when it writes the value 99.

### PropagateRequiresGradFlowsForwardThroughTheWholeChain

Before `finalise`, the MATMUL of an input and a parameter and the ReLU after it do not
require a gradient. After it the input still does not, and the parameter, the product and the
ReLU do.

### PropagateRequiresGradLeavesAFrozenBranchAlone

After `finalise`, a ReLU of the input does not require a gradient and the MATMUL of the input
and a parameter does.

### BuilderInfersMatmulShapesForEveryTranspositionCase

With [4, 3] or [3, 4] on the left and [3, 5] or [5, 3] on the right, `matmul` gives [4, 5]
for each of the four combinations of the two transpose flags.

### BuilderRejectsAContractionMismatch

`matmul` of [4, 3] and [7, 5] throws `Error`.

### BuilderChecksBiasAndLossShapes

Over a [4, 5] value, `bias_add` with a [5] bias gives [4, 5] and with a [4] bias throws
`Error`. `sum_rows` gives [5]. `nll_loss` with [4, 5] targets gives [1] and with [4, 6]
targets throws `Error`.

### BuilderRejectsMixedElementTypes

`add` and `matmul` of a `Float32` and a `Float64` operand both throw `Error`.

### NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

### InsertNodeRejectsAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

`node_count()` is still 1 after the throw. The insert was rejected, not half-applied: nothing
was inserted and no producer was claimed.

The final `EXPECT_NO_THROW` inserts the same node ahead of that reader and succeeds. That makes
the rejection a statement about position and not about the node.

### InsertNodeRejectsAnOperandProducedNoEarlierThanItself

With one RELU node writing value 1, `insert_node` at 0 of a TANH that reads value 1 raises an
error holding "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes" and "topological".
The graph still has one node and the TANH's output has no producer.

### AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::K_CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
the lowering reads each input through `LoweringContext::input` and passes the tensor straight to
SushiBLAS.

### ABufferLeafIsPersistentAndNeedsNoGradient

The nine cases from here are in the suite `Unit_Ir`; the cases above are in `Unit_IR`.

`GraphBuilder::buffer` declares a value of the shape, type and name it was given, with the
lifetime `Lifetime::Buffer` and `requires_grad` false. Its producer is a K_LEAF node. A K_BIAS_ADD
that reads an input and a buffer does not require a gradient either. `is_persistent(BUFFER)` is
true and `lifetime_name(BUFFER)` is "buffer".

### ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist

`Graph::is_persistent` is true for PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT and BUFFER, and
false for ACTIVATION and SCRATCH.

### StateLeafReturnsTheBoundLeafAndNoValueForAnUnboundState

`Graph::state_leaf` returns `K_NO_VALUE` before any binding. After `bind_state_leaf` of state 7
to a buffer and state 5 to a parameter, it returns each leaf for its state and `K_NO_VALUE` for
state 6 and for `K_NO_STATE`. `state_leaves()` holds the two pairs in the order they were
bound, not in the order of the state numbers.

### BindStateLeafRefusesASecondBindingOfAStateOrOfALeaf

With state 7 bound to a leaf, four bindings throw: state 7 to another leaf ("State 7",
"already has leaf"), state 8 to the bound leaf ("already stands for state 7"), `K_NO_STATE` to
a leaf ("no identity"), and state 9 to value 99 ("value 99", "does not exist"). The table
still holds the first pair alone.

### WithoutNodesKeepsEveryValueAndStateLeafAndHoldsNoNode

`Graph::without_nodes()` returns a graph with no node and the same number of values. Every
value keeps its shape, element type, layout, lifetime, `requires_grad` and debug name, and its
producer is `K_NO_NODE`. The fixture holds an NCHW value, a parameter that requires a gradient
and a HALF buffer, and each is checked by name. The copy's `state_leaves()` equals the
source's. The source graph still has its nodes and producers.

### ARefusalIsReportedAfterItIsMarkedAndSurvivesWithoutNodes

`has_refusal()` is false for a new graph and for its `without_nodes()` copy. After
`mark_refused()` it is true for both.

### StateIdentityDrawsOneIdentityPerHolder

`Graph::StateIdentity` in `modules/core/graph/include/SushiAI/graph/state_id.hpp` holds the identity of one piece
of model state. `peek()` returns `K_NO_STATE` before the first `get()`. `get()` returns the
same identity on every call, `peek()` then returns it too, and a second holder's `get()`
returns a different one.

### ACopiedStateIdentityDoesNotShareTheSourcesIdentity

A holder copy-constructed from one that has drawn peeks `K_NO_STATE` and then draws a
different identity. A holder that had drawn and is then copy-assigned from the source also
peeks `K_NO_STATE`, and its next identity differs from the source's and from the one it had.
The source keeps its identity.

### AMovedStateIdentityCarriesItsIdentityAway

After a move construction the target peeks the source's identity and the source peeks
`K_NO_STATE`. A move assignment over a holder that had drawn does the same.

## test_memory_plan.cpp

Suites: `Unit_MemoryPlan`.

Tests that the memory planner never shares a byte between two values live together.

This file needs no device: no engine, no allocator, no SYCL. The memory planner is a pure
function of the IR, which is the reason the IR is a host-lane structure, and every case is a
synthetic graph and a table of offsets.

The contract is one sentence: two values whose live ranges intersect never share a byte, and
two whose live ranges do not are free to. The first half is checked exhaustively over every
pair. The liveness it is checked against is computed in the test file by a breadth-first search,
`descendants()`, and not by asking the planner what it thinks. A planner checked against its own
reachability code would agree with itself whatever it did.

The individual cases are for the second half, and the diamond in
TwoUnorderedBranchesNeverShareBytes is the one that matters. It is the shape where a planner
that overlaps disjoint [first_use, last_use] intervals on a linearised schedule gets a wrong
answer that no test written from the linear model would catch.

The cases fall into six groups, in file order:

- degenerate graphs
- the two shapes that decide whether the pass is right
- lifetimes
- aliasing
- mixed dtypes
- knobs, refusals and malformed input

### AnEmptyGraphAsksForNothing

The plan of an empty graph has 0 values, no arena and 0 arena bytes.

### ASingleActivationGetsOneArenaAndOffsetZero

For one ReLU over an input, the plan has one arena of the `Activation` lifetime with one
buffer and 128 bytes. The input is not packed. The ReLU's value is packed at offset 0 with
128 bytes.

### AGraphOfOnlyLeavesPacksNothing

The plan of a graph of one input and one parameter has no arena and 0 arena bytes.

### AChainPingPongsBetweenTwoBuffers

The graph is x -> a -> b -> c -> d. Nothing branches, so `a` is finished before `c` starts and
`b` before `d`. Four activations need exactly two buffers, which is the most any planner can do
with a chain.

### TwoUnorderedBranchesNeverShareBytes

This is the case the planner is judged by. Linearised, the two branches have disjoint
[first, last] intervals: a1 dies at node 2 and b1 is born at node 3. An interval planner
overlaps them and is wrong. Nothing in the graph orders node 2 before node 3 and the runtime may
run the branches concurrently, so the sharing would corrupt the result and also serialise the
parallelism it was meant to leave free.

### OnlyTheNonPersistentClassesArePacked

The graph holds all six lifetimes. The four persistent ones live for the whole run, so packing
them would save nothing and would merge the dependency-tracker shards that matter most. Only
ACTIVATION and SCRATCH reach an arena, and they reach different ones.

### AValueNothingWritesKeepsItsOwnAllocation

An ACTIVATION with no producer is written from outside the step and never rewritten, so its
bytes are live for the whole run whatever its class says. Handing them away would survive
exactly one replay.

### AnAccumulateChainIsOneBufferWithOneLiveRange

K_ACCUMULATE's result is its accumulator's storage. Planning the two separately would be silently
wrong and not merely wasteful: the planner would think the accumulator died at the K_ACCUMULATE
node and hand its bytes to `sink`, which is produced by the node still reading them.

### EveryStartIsAWholeNumberOfItsOwnElements

Alignment is turned down to 1 so that the element-width floor is the only thing left doing the
work. A 20-byte float32 buffer at offset 0 leaves the next free byte at 20, which is not a whole
number of float64 elements. The float64 buffer therefore has to start at 24, or its
storage_offset would not be representable at all.

### ReuseCanBeSwitchedOffAndSaysSo

With `reuse` false over a chain of three ReLUs, the plan reports reuse disabled, a non-empty
`reuse_note` and 384 arena bytes.

### TooManyNodesDeclinesOverlapRatherThanApproximatingIt

The refusal is the point. A planner that quietly fell back to overlapping linear intervals
would produce a plan that is wrong in the way TwoUnorderedBranchesNeverShareBytes describes, and
would report a saving for it.

### APinnedValueKeepsItsOwnAllocation

With the first of two chained ReLU values pinned, that value is not packed and the second is.

### PinningAValueThatDoesNotExistThrows

`MemoryPlan::build` with the value 99 pinned throws `Error`.

### ZeroAlignmentThrows

`MemoryPlan::build` with an alignment of 0 throws `Error`.

### AGraphThatIsNotInTopologicalOrderThrows

GraphBuilder cannot produce this graph, because it always emits a producer before anything can
read it. The whole pass rests on the property all the same, so the test checks it. A closure
built over a back edge would silently miss ordering and the plan would overlap things it must
not.

### ThePlanIsDeterministicForTheSameGraph

The offsets are byte-identical across two builds. A plan that shuffled would move every tensor
and take the library's bit-for-bit replay guarantee with it.

### TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes

`FusedForm::GEMMBiasGELU` writes two values from one node: its result and the pre-activation
its epilogue spills. They have the same shape, they are born at the same instant, and the
planner packs by liveness. "One died before the other was written" is the mistake available to
it, and taking it would have the GELU computed from its own output with nothing to show for it.

Nothing special prevents it, which is why it is worth pinning. `precedes()` asks whether every
use of the earlier buffer strictly reaches the later one's first write, and a node does not
reach itself. The planner therefore refuses both directions and the two values conflict. That is
an inference from an existing rule and not a rule of its own, so a future rewrite of
`precedes()` could take it away silently.

## test_step_recorder.cpp

Suites: `Unit_StepRecorder`, `Unit_StepRecorderDeathTest`.

Tests that StepRecorderList drives its members as one recorder, in the order added.

This file needs no device. `Graph::StepRecorder` and `Graph::StepRecorderList` are in
`modules/core/graph/include/SushiAI/graph/step_recorder.hpp`. The list is itself a `StepRecorder` and holds
references to its members; it does not own them. `add` returns a Status, and each case checks
its call with `SA_EXPECT_OK`.

### JournalRecorder

`JournalRecorder` appends "<name>.record", "<name>.before" or "<name>.after" to a journal the
test owns, and `record_into_step()` returns the task count given at construction.

### RecordOnlyRecorder

`RecordOnlyRecorder` overrides `record_into_step()` alone and returns 3. Its `before_replay()`
and `after_replay()` are the interface's own.

### in_order_added

`in_order_added` returns the three journal entries that recorders named a, b and c write for
one call.

### EmptyListRecordsNothing

An empty list reports `empty()`, returns 0 from `record_into_step()` and does not throw from
either replay hook.

### AddMakesTheListNonEmpty

After one `add` the list is not empty and the journal is still empty, so `add` calls nothing
on the member.

### RecordSumsTheMemberCounts

Members that report 2, 0 and 5 tasks give `record_into_step()` 7.

### RecordCallsMembersInTheOrderAdded

`record_into_step()` over a, b and c journals "a.record", "b.record", "c.record".

### OrderFollowsAddNotConstruction

With b added before a, the journal is "b.record", "a.record", although a was constructed
first.

### BeforeReplayCallsMembersInTheOrderAdded

`before_replay()` over a, b and c journals "a.before", "b.before", "c.before".

### AfterReplayCallsMembersInTheOrderAdded

`after_replay()` over a, b and c journals "a.after", "b.after", "c.after". The members are not
called in reverse.

### MemberWithoutReplayHooksIsDriven

A list of a `RecordOnlyRecorder` and a journal recorder of 4 tasks records 7. After both replay
hooks the journal holds "a.record", "a.before" and "a.after" only.

### ListIsDrivenThroughTheInterface

Through an `StepRecorder*`, a list of a and b records 5 tasks and journals the record calls,
then the before calls, then the after calls, each as a then b.

### NestedListKeepsItsPlaceInTheOrder

An outer list of a, an inner list holding b, and c records 7 tasks and journals "a.record",
"b.record", "c.record".

### AddRejectsARecorderAlreadyHeld

A second `add` of the same recorder is a programmer's misuse, so it stops the process with a report
containing "StepRecorderList::add" and "already". The case runs as a death test in
`Unit_StepRecorderDeathTest`. The parent process then checks that the list still records that
member once: 2 tasks and one journal entry.

### AddRejectsTheListItself

`list.add(list)` stops the process with a report containing "StepRecorderList::add" and "itself".
It is a death test in `Unit_StepRecorderDeathTest`, and the list stays empty.

## test_traced.cpp

Suites: `Unit_Traced`.

Tests Traced and the refusal it carries through a chain of emitters.

This file needs no device. `Graph::Traced` in `modules/core/graph/include/SushiAI/graph/traced.hpp` holds a
`ValueID` or a failed `Status`. An emitter that returns one takes its operands as `Traced`
too, so a call can be nested inside another and the first refusal travels to the outermost
result.

Six `static_assert` lines at the top of the file hold that a `Traced` is default-constructible
and constructible from a `ValueID` and from a `Status` without throwing, that both conversions
are implicit, and that its size is two pointers.

The cases up to BuildingFromASuccessStatusIsFatal use `StandInBuilder` and involve no graph.
The last four use a `Graph::GraphBuilder` and read `Graph::has_refusal()`.

### TraceLog

`TraceLog` counts the values the stand-in builder emitted and the refusals it recorded.

### StandInBuilder

`StandInBuilder` has two members. `emit()` returns the next id, and `refuse()` counts one
refusal and returns the failure it was given. `SA_REFUSE_IF` is run over it in `emit_below`.

### emit_below

`emit_below` binds its operand through `SA_TRY_ASSIGN`, refuses through `SA_REFUSE_IF` when
the id is above a limit, and otherwise emits one value.

### emit_over

`emit_over` binds its operand through `SA_TRY_ASSIGN` and emits one value. It has no condition
of its own, so it accepts the absent operand.

### forward_traced

`forward_traced` runs `SA_TRY` on a `Traced` and sets a flag after it.

### declare_below

`declare_below` runs `SA_REFUSE_IF` over a `Graph::GraphBuilder` when an extent is above a
limit, and otherwise declares one input of that extent.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### DefaultHoldsTheAbsentValueAndNoFailure

A default-constructed `Traced` has a value, an ok status and the id `Graph::K_NO_VALUE`.

### HoldsAnIDWithASuccessStatus

`Traced(ValueID{12})` has a value, is ok and returns 12.

### HoldsAFailureAndTheAbsentID

A `Traced` built from a failure has no value and reports the failure's code and message.

### SuccessPathCreatesNoFailureBlock

Two chained emitters over an absent operand, a copy and a move yield the id 1, record no
refusal and leave `live_failure_count()` where it was.

### MovedFromTracedKeepsItsState

A failed `Traced` that was moved from still reports its failure, as does the target. A
`Traced` holding 4 that was move-assigned from still returns 4, as does the target.

### AssignmentReplacesAFailureWithAnID

Assigning `ValueID{3}` to a failed `Traced` leaves a value and the id 3. Assigning a failure to
it afterwards leaves that failure.

### AnAbsentOperandPassesThroughAnEmitter

`emit_over(builder, Traced())` succeeds with the id 0, and the log shows one emitted value.

### RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing

`emit_below` given the id 9 and the limit 4 fails with "operand 9 is above the limit 4",
records one refusal and emits nothing. That result is then passed through `emit_below` with
the limit 100, nested inside `emit_over`. The outer result carries the same message, and its
`message().data()` is the first result's pointer, so the failure was forwarded and not rebuilt.
The log still shows one refusal and no emitted value.

### NestedCallsEmitWhenNothingIsRefused

`emit_over(builder, emit_below(builder, ValueID{2}, 4))` yields the id 1 with two values
emitted and no refusal.

### TryForwardsATracedFailureAsAStatus

`forward_traced` over a failed `Traced` returns the failure and leaves its flag false. Over
`ValueID{1}` it succeeds and sets the flag.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Traced` stops the process with a report containing "Traced::value() was
called on a failure" and the failure's message.

### BuildingFromASuccessStatusIsFatal

`Traced(Status())` stops the process with "a Traced was built from a success Status".

### RefuseMarksTheGraphAndReturnsTheFailure

`GraphBuilder::refuse` returns the failure it was given, with the same message pointer, and
`graph.has_refusal()` goes from false to true.

### RefuseIfOverAGraphBuilderMarksTheGraphOnlyOnARefusal

`declare_below(builder, 4, 8)` declares value 0 and leaves the graph without a refusal.
`declare_below(builder, 9, 8)` fails with "extent 9 is above the limit 8", adds no value and
marks the graph.

### ObserveMarksTheGraphForAFailedStatusOnly

`GraphBuilder::observe(Status())` is ok and leaves the graph unmarked. `observe` of a failed
status returns that failure and marks the graph.

### ObserveMarksTheGraphForAFailedResultOnly

`observe(Result<int>(7))` returns a result holding 7 and leaves the graph unmarked. `observe`
of a failed `Result<int>` returns the failure and marks the graph.

## test_builder.cpp

Suites: `Unit_BatchedMatmul`.

Tests the IR that `GraphBuilder::matmul` emits for rank-2 and rank-3 operands and that
`GraphBuilder::softmax` emits over the last axis. Every refusal case also expects the graph's
node count to be what it was before the refused calls.

### Rank2ProductKeepsItsShapeAndFlags

`matmul` over [3, 4] and [5, 4] with the second operand transposed yields [3, 5]. The node's
operation is `Ops::K_MATMUL`, its flag 0 is false and its flag 1 is true.

### FourFlagCombinationsYieldBatchedProducts

For each of the four transpose pairs, with operands shaped to fit it, a rank-3 product yields
[2, 3, 5]. The node is a MATMUL that reads the two operands in order, writes one value and
carries the two flags it was given.

### BatchedProductTakesTheSharedBatchCount

`matmul` over [7, 2, 3] and [7, 3, 6] yields [7, 2, 6] with the first operand's layout and the
element type `Float32`.

### RefusesMixedRanksAndRanksOutsideTwoAndThree

`matmul` throws `Error` for a rank-2 operand with a rank-3 one in either order, for a rank-1
operand with a rank-3 one, and for two rank-4 operands.

### RefusesUnequalBatchesAndInnerExtents

Against a [2, 3, 4] operand, `matmul` throws `Error` for a [3, 4, 5] operand, whose batch
differs, and for a [2, 5, 3] operand, whose inner extent differs.

### SoftmaxEmitsReshapeLogSoftmaxExpReshape

`softmax` over a [2, 3, 3] input adds four nodes. Read back from the result they are RESHAPE,
EXP, LOG_SOFTMAX and RESHAPE. The first RESHAPE reads the input and yields [6, 3], LOG_SOFTMAX
yields [6, 3], and the result has the shape [2, 3, 3].

### SoftmaxOverRank2KeepsItsRowsAndLastAxis

`softmax` over a [4, 5] input yields [4, 5], and the EXP node behind the last RESHAPE reads a
[4, 5] value.

### SoftmaxRefusesRankBelowTwo

`softmax` over a [5] input throws `Error` and adds no node.

Suites: `Unit_CausalMaskOps`.

Tests `GraphBuilder::causal_mask`: the node it emits, the shapes it accepts and the operands it
refuses. Every case uses the fill -1e9.

### TheMaskWritesItsIdentityAndKeepsTheShape

Over [2, 5, 5] float64 scores the call is ok and emits one CAUSAL_MASK node that reads the
scores, writes the returned value and holds the fill in attribute slot 0. The result has the
shape [2, 5, 5], the type float64 and the lifetime `Activation`, and the graph has no refusal.

### TheMaskAcceptsAnyNumberOfLeadingAxes

A [3, 3] matrix and a [2, 4, 3, 3] value both return a value of their own shape.

### TheMaskRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: a [5] vector ("[..., T, T] scores"), a [2, 4, 5]
value ("[2, 4, 5]"), int32 scores ("half, float32 or float64 scores") and a fill of minus
infinity ("finite fill"). After each the node count is unchanged and `has_refusal()` is true.

Suites: `Unit_DepthwiseOps`.

Tests the three depthwise emitters of `GraphBuilder`: `depthwise_conv2d`,
`depthwise_conv2d_backward_input` and `depthwise_conv2d_backward_weight`. Most cases share one
geometry, a 3x3 window with unit stride and one pixel of padding, over a [2, 5, 5, 3] input and
[3, 3, 3] taps.

### DepthwiseConv2dWritesItsIdentityOverInputAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D` node that reads the input and then the
taps and writes that one value.

### DepthwiseConv2dYieldsTheWindowGridOverThePaddedInput

The shared geometry yields [2, 5, 5, 3]. The same window with stride 2 on both axes yields
[2, 3, 3, 3].

### DepthwiseConv2dKeepsLayoutDTypeAndLifetime

Over a `Float64` input and `Float64` taps the result is tagged NHWC, has the element type
`Float64` and the lifetime `Activation`.

### DepthwiseConv2dStoresItsGeometryInTheGeometrySlots

For a 3x5 window with padding 1 on top and 2 on the left over a [2, 7, 9, 3] input, the node's
first eight attribute values are 3, 5, 1, 1, 1, 1, 1 and 2.

### DepthwiseConv2dRefusesARankFourWeight

Taps of the shape [3, 3, 3, 1] make the emitter throw `Error`, and the node count does not
change.

### DepthwiseConv2dRefusesTapsOfAnotherWindowOrChannelCount

Taps of the shapes [5, 3, 3], [3, 5, 3] and [3, 3, 4] each make the emitter throw `Error`
against the 3x3 geometry and three channels, and the node count does not change.

### DepthwiseConv2dRefusesAnNCHWInput

An input converted to NCHW with `layout_convert` makes the emitter throw `Error`.

### DepthwiseConv2dRefusesTapsOfAnotherDType

A `Float32` input with `Float64` taps makes the emitter throw `Error`.

### DepthwiseConv2dRefusesAWindowThatDoesNotFit

A 3x3 window without padding over a [2, 2, 2, 3] input makes the emitter throw `Error`.

### BackwardInputWritesItsIdentityOverGradientAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT` node that reads the
gradient and then the taps.

### BackwardInputRecoversTheForwardInputShape

From a [2, 3, 3, 3] gradient, stride 2 and the input shape [2, 5, 5, 3], the result has the
shape [2, 5, 5, 3], the layout NHWC and the element type `Float32`.

### BackwardInputRefusesAGradientOnAnotherGrid

For the input shape [2, 5, 5, 3], gradients of the shapes [2, 4, 5, 3], [2, 5, 5, 4] and
[2, 5, 3] each make the emitter throw `Error`, and the node count does not change.

### BackwardWeightWritesItsIdentityOverGradientAndInputInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT` node that reads the
gradient and then the forward input.

### BackwardWeightYieldsTheTapShapeInTheOperandsDType

Over `Float64` operands the result has the shape [3, 3, 3], the layout NHWC, the element type
`Float64` and the lifetime `Activation`.

### BackwardWeightRefusesMismatchedBatchChannelsOrDType

Against a [2, 5, 5, 3] gradient, an input of another batch, of another channel count or of
another element type makes the emitter throw `Error`, as does a rank-3 gradient. The node count
does not change.

Suites: `Unit_ElementwiseOps`.

Tests the nodes the builder emits for the elementwise, comparison, selection and constant
operations.

This file needs no device. Every case in `Unit_ElementwiseOps` calls an emitter of
`modules/core/graph/include/SushiAI/graph/builder.hpp` on a host-side graph and reads the node and the value it
wrote. The emitters are `div`, `minimum`, `maximum`, `less_equal`, `abs`, `atan`, `softplus`,
`silu`, `select`, `activation_backward`, `constant` and `constant_like`. The cases that concern
several emitters loop over a table and label each iteration with `SCOPED_TRACE`, so a failure
names the emitter. `test_backward.cpp` covers the backward rules of the same operations.

### BinaryEmitter and UnaryEmitter

`BinaryEmitter` holds a label, the identity an emitter must write and a pointer to a
two-operand member of `GraphBuilder`. `UnaryEmitter` is the same for one operand.

### K_BINARY_EMITTERS

`K_BINARY_EMITTERS` lists `div`, `minimum`, `maximum` and `less_equal` with K_DIV, K_MINIMUM,
K_MAXIMUM and K_LESS_EQUAL.

### K_UNARY_EMITTERS

`K_UNARY_EMITTERS` lists `abs`, `atan`, `softplus` and `silu` with K_ABS, K_ATAN, K_SOFTPLUS and K_SILU.

### K_INPUT_READING_GRADIENTS

`K_INPUT_READING_GRADIENTS` lists K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, the two activation
gradients that read the forward input.

### nchw_input

`nchw_input` declares a [2, 2, 2, 2] input and converts it to NCHW. Every extent is 2, so the
permuted shape equals the original and two such values still agree in shape.

### expect_produced_by

`expect_produced_by` expects a value to be the single result of a node of a given operation
that reads a given operand list in order.

### expect_described_like

`expect_described_like` expects a value to have the shape, element type and layout of another
value and the lifetime ACTIVATION.

### BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder

Each two-operand emitter writes one node of its own identity whose inputs are `a` then `b`.

### BinaryEmittersKeepShapeDTypeAndLayout

With two FLOAT64 NCHW operands, the result of each two-operand emitter has the first operand's
shape and is FLOAT64 and NCHW.

### BinaryEmittersRejectUnequalShapesWithoutBroadcasting

Each two-operand emitter throws for a [4, 3] operand with a [4, 1] one, in both orders, and
with a [3] one. These are the pairs a broadcasting library accepts. The node count is unchanged
after the three calls.

### BinaryEmittersRejectUnequalDtypes

Each two-operand emitter throws for a FLOAT32 operand with a FLOAT64 one, in both orders.

### LessEqualYieldsAMaskInTheOperandsOwnDType

The result of `less_equal` is FLOAT32 for FLOAT32 operands and FLOAT64 for FLOAT64 ones. The
mask has no element type of its own, which is what lets `mul` and `select` read it beside a
value of the same type.

### UnaryEmittersWriteTheirIdentityOverTheirOperand

Each single-operand emitter returns a value other than its operand, written by one node of its
own identity that reads the operand.

### UnaryEmittersKeepShapeDTypeAndLayout

With a FLOAT64 NCHW operand, the result of each single-operand emitter has the operand's shape
and is FLOAT64 and NCHW.

### SelectReadsItsConditionThenBothBranches

`select(condition, a, b)` writes one K_SELECT node whose inputs are the condition, `a` and `b`,
in that order.

### SelectKeepsShapeDTypeAndLayout

With three FLOAT64 NCHW operands the result has the shape, element type and layout of `a`.

### SelectRejectsAConditionOfAnotherShape

A [4, 1] condition over [4, 3] branches throws, and the node count is unchanged.

### SelectRejectsAConditionOfAnotherDType

An INT64 condition over FLOAT32 branches throws.

### SelectRejectsBranchesOfUnequalShape

A [4, 3] branch with a [4, 1] branch throws in both orders.

### SelectRejectsBranchesOfUnequalDType

A FLOAT32 branch with a FLOAT64 branch throws in both orders.

### ActivationBackwardEmitsTheInputReadingGradients

For K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, `activation_backward(gradient, dy, x)` writes one node
of that identity whose inputs are `dy` then `x`, and the result is described like `dy`.

### InputReadingGradientsRejectAGradientOfAnotherShape

A [4, 1] `dy` with a [4, 3] `x` throws for both gradients.

### InputReadingGradientsRejectAGradientOfAnotherDType

A FLOAT32 value with a FLOAT64 one throws for both gradients, in both orders.

### ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient

`activation_backward` throws when the identity is K_SILU, K_SOFTPLUS or K_DIV. The first two are
forward identities of `Ops::K_ACTIVATIONS` and the third is not in the table.

### FinaliseAcceptsAGraphUsingEveryEmitter

Two NCHW inputs are chained through all four two-operand emitters and all four single-operand
ones, and the chain ends in a `select` on a `less_equal` mask. `Graph::finalise()` does not
throw.

### FinaliseRejectsOperandsInTwoLayouts

For each two-operand emitter, a node that reads one NCHW operand and one NHWC operand is
emitted, and `finalise()` throws. The emitter itself accepts the pair because the shapes are
equal.

### FinaliseRejectsASelectConditionInAnotherLayout

A `select` with an NHWC condition over NCHW branches is emitted, and `finalise()` throws.

### ResultsOfAParameterRequireAGradient

After `finalise()`, `silu(div(weight, constant))` requires a gradient and
`atan(maximum(constant, constant))` over an input does not.

### AComparisonOfAParameterRequiresNoGradient

After `finalise()`, the mask `less_equal(weight, bound)` requires no gradient, and neither does
a `select` that reads it as its condition over two inputs.

### ConstantWritesAnOperandFreeNodeHoldingItsValue

`constant([4, 3], 1e-7, FLOAT64)` writes one K_CONSTANT node with no input. Attribute slot 0
holds 1e-7 within 1e-20, and the value is [4, 3], FLOAT64, NHWC and ACTIVATION. `K_VALUE` and
`K_VALUE_TOLERANCE` are constants local to the case.

### ConstantDefaultsToFloat32

`constant([2], 1.0)` with no element type is FLOAT32.

### ConstantRejectsAnElementTypeThatIsNotRealFloating

`constant` throws for INT64 and for COMPLEX32, and the node count and the value count are
unchanged.

### ConstantLikeCopiesShapeDTypeAndLayoutWithoutReadingItsOperand

`constant_like(x, 0.5)` over a FLOAT64 NCHW value writes a K_CONSTANT node with no input whose
result is described like `x`. The value it names is not an operand of the node.

### ConstantLikeRejectsAValueThatIsNotRealFloating

`constant_like` over an INT64 value throws, and the node count and the value count are
unchanged.

### AConstantLikeAValueIsAnOperandThatFinaliseAccepts

`add(x, constant_like(x, 1e-7))` over an NCHW input passes `finalise()`, and the sum requires
no gradient.

### ZerosStillWritesTheZeroIdentity

`zeros` and `zeros_like` each write a K_ZERO node with no input. K_ZERO and K_CONSTANT are two
identities.

Suites: `Unit_GatherOps`.

Tests `GraphBuilder::gather_rows` and `GraphBuilder::scatter_add_rows`: the node each emits,
the result shape and the operands each refuses. The table has 7 rows of length 3. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

### GatherRowsWritesItsIdentityOverTheTableAndTheIndex

Over a [7, 3] table and a [5] int32 index the call is ok and the result is the one output of a
GATHER_ROWS node that reads the table and then the index. The result has the shape [5, 3], the
type float32 and the lifetime `Activation`.

### GatherRowsAppendsTheRowLengthToAnIndexOfAnyRank

Over a float64 table, a [2, 4] index gives a float64 [2, 4, 3] value and a [2, 1, 3, 1, 2]
index gives a [2, 1, 3, 1, 2, 3] value.

### GatherRowsRefusesATableThatIsNotRankTwo

A [7] table is refused with "rank-2 [V, D] table" and a [7, 2, 3] table with "[7, 2, 3]".

### GatherRowsRefusesAnIndexThatIsNotInt32

A float32 index and an int64 index are each refused with "INT32 index".

### GatherRowsRefusesATableThatIsNotFloating

An int32 table is refused with "half, float32 or float64 table".

### GatherRowsRefusesAnIndexWhoseResultWouldPassTheHighestRank

An index of rank `K_MAX_RANK`, shaped [1, 1, 1, 1, 1, 1], is refused with "rank 1 to 5".

### ScatterAddRowsWritesItsIdentityAndTheTableShape

Over [2, 4, 3] float64 rows and a [2, 4] index, `scatter_add_rows` with 7 table rows is ok. The
result is the one output of a SCATTER_ADD_ROWS node that reads the rows and then the index, and
it has the shape [7, 3], the type float64 and the lifetime `Activation`.

### ScatterAddRowsRefusesOperandsThatDoNotFit

Five calls are refused: rank-1 rows ("rank 2 or more"), int32 rows ("half, float32 or float64
rows"), a float32 index ("INT32 index"), a [4] index for five rows ("one index per row") and a
row count of 0 ("a table of 0 rows").

### AnInt32InputLeafRequiresNoGradient

After `graph.finalise()` the int32 index has the lifetime `Input` and does not require a
gradient, and the gathered value does.

### TheCastPassLeavesAnInt32LeafAndItsGatherAsDeclared

After `Graph::insert_casts` under a `MixedPrecisionPolicy`, the index is int32, the table and
the gathered value are float32 and the `matmul` that reads the gathered value is `Half`. The
gather still reads the table and the index. No reported cast has the index as its source or
GATHER_ROWS as its op, and `find_dtype_mismatches` is empty.

Suites: `Unit_LayerNormOps`.

Tests `GraphBuilder::layer_norm` and `GraphBuilder::layer_norm_backward`. The leaves are a
[2, 3, 4] input, a [4] gain and a [4] bias, and the epsilon is 1e-5. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

### LayerNormWritesItsValueAndTheTwoRowStatistics

Over float64 leaves the call is ok and emits one LAYER_NORM node that reads x, the gain and the
bias and writes the output, the saved mean and the saved inverse deviation, with the epsilon in
attribute slot 0. The output has the shape [2, 3, 4] and both saves the shape [6]. The output
and the saved mean are float64, the output has the lifetime `Activation` and the graph has no
refusal.

### ARankOneInputIsOneRow

Over a [4] input the output has the shape [4] and the saved mean the shape [1].

### LayerNormRefusesOperandsThatDoNotFit

Four calls are refused: an int32 input ("half, float32 or float64 input"), a [3] gain ("gamma
as [4]"), a float64 bias under a float32 input ("beta as [4] of float32") and an epsilon of 0
("epsilon must be positive").

### LayerNormBackwardWritesTheThreeGradients

`layer_norm_backward` is ok and emits one LAYER_NORM_BACKWARD node that reads dy, x, the gain,
the saved mean and the saved inverse deviation and writes dx, dgamma and dbeta, of the shapes
[2, 3, 4], [4] and [4].

### LayerNormBackwardRefusesOperandsThatDoNotFit

A [6, 4] gradient is refused with "operand of [2, 3, 4]" and a [4] saved mean with "operand of
[6]".

Suites: `Unit_LogSoftmaxOps`.

Tests `GraphBuilder::log_softmax_backward`.

### TheGradientWritesItsIdentityAndKeepsTheShape

Over a [4, 3] float64 gradient and the `log_softmax` of a [4, 3] float64 input, the call is ok
and emits one LOG_SOFTMAX_BACKWARD node that reads the gradient and then the log-probabilities.
The result has the shape [4, 3], the type float64 and the lifetime `Activation`, and the graph
has no refusal.

### TheGradientRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: rank-3 operands ("rank-2 log-probabilities"), int32
operands ("half, float32 or float64 log-probabilities"), a [4, 5] gradient ("[4, 5]") and a
float64 gradient under float32 log-probabilities ("a gradient like them"). After each the node
count is unchanged and `has_refusal()` is true.

## lowering/test_random.cpp

Suites: `Unit_DropoutOps`.

Tests `GraphBuilder::dropout`: the node it emits, the seed it stores, the operands it refuses
and its row in the lowering table. The accepted cases pass the rate 0.25.

### DropoutWritesItsValueAndTheMaskItSaves

Over a [2, 3, 4] float64 input the call is ok and emits one DROPOUT node that reads the input
and writes the output and then the saved mask. The node holds 0.25 in `K_DROPOUT_RATE_SLOT`.
Both outputs have the shape [2, 3, 4], the type float64 and the lifetime `Activation`, and the
graph has no refusal.

### TheSiteSeedSurvivesItsAttributeSlots

For the seeds 0xFEDCBA9876543210, 0 and all 64 bits set, `Graph::dropout_seed` of the emitted
node's attributes returns the seed that was passed.

### DropoutRefusesOperandsThatDoNotFit

An int32 input fails with `InvalidDescription` and "half, float32 or float64 input". The rates
0, 1 and -0.5 each fail with "rate in (0, 1)". After each the node count is unchanged and
`has_refusal()` is true.

### DropoutHasTheRowOfTheRandomTable

`Graph::K_RANDOM_LOWERINGS` has one row, for DROPOUT. `find_lowering` returns
`Graph::lower_dropout`, `lowering_row_count` is 1 and `is_unlowered` is false.

## lowering/test_all.cpp

Suites: `Unit_LoweringTable`.

Tests the table that maps each operation identity to its lowering. The lowering is eight domain
tables under `modules/core/graph/include/SushiAI/graph/lowering/`, which `all.hpp` joins into `K_LOWERINGS`. The
identities that have no row are listed in `K_UNLOWERED`, in `unlowered.hpp`.

The file needs no device. It reads the constants and the constexpr helpers of `all.hpp`. The
`static_assert`s in `all.hpp` prove the same counts at build time. These cases name the identity
or the domain whose count is wrong, so a change to one table reads as one line in the report.

### EveryIdentityIsLoweredOrListedAsUnlowered

Each entry of `Ops::K_ALL` has exactly one row, or it is listed in `K_UNLOWERED` and has none.

### TheUnloweredIdentitiesHaveNoRow

`K_UNLOWERED` holds four identities, and none of them has a row.

### NoOperationHasTwoRows

No identity appears in two rows of `K_LOWERINGS`.

### TheCombinedTableCoversTheOperationTable

Every identity is accounted for, and the rows plus the unlowered identities add up to the
operation table's size.

### EachDomainTableHasItsRowCount

The domain tables hold 17, 14, 6, 3, 4, 15, 6 and 6 rows, in the order elementwise, activation,
linalg, loss, shape, spatial, norm and fused. The joined table holds 71.
