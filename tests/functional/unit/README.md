# Unit tests

The unit tier runs without a device. Every file in this folder builds a graph, a config document
or a byte stream on the host and inspects the result, so no case needs an Engine, a TensorPool or
SYCL.

This page holds the reasoning behind the cases. A `//` line in a test states the invariant the
case relies on and cites this page. The section under the test's name says why the invariant
matters, which number it was measured at, or which failure it once let through. Where a test has
several such lines, each paragraph opens with the assertion or the statement it explains.
Sections follow the files in path order, and inside a file the headings follow the order of the
tests.

## Files

| File | What it tests |
| --- | --- |
| [`test_backward.cpp`](test_backward.cpp) | The graph that differentiation emits and the coverage of the backward rule table. |
| [`test_checkpoint_format.cpp`](test_checkpoint_format.cpp) | The checkpoint format's round trip and its refusal of each malformed byte pattern. |
| [`test_config.cpp`](test_config.cpp) | How the config layer parses its JSON documents and which ones it refuses. |
| [`test_diagnostics.cpp`](test_diagnostics.cpp) | That each host-lane precondition raises a message carrying the values that disagree. |
| [`test_fusion.cpp`](test_fusion.cpp) | The matches and the refusals of the static fusion pass on synthetic graphs. |
| [`test_gradcheck.cpp`](test_gradcheck.cpp) | The gradient checker against closed-form functions and a wrong gradient. |
| [`test_ir.cpp`](test_ir.cpp) | The IR's single-assignment and ordering rules and the builder's shape inference. |
| [`test_layer_stack.cpp`](test_layer_stack.cpp) | That a LayerStack names and declares its parameters exactly as Sequential does. |
| [`test_layout.cpp`](test_layout.cpp) | That every value carries a tensor layout and that the rebuilding passes keep it. |
| [`test_loss_scaling.cpp`](test_loss_scaling.cpp) | The dynamic loss-scaling rule on LossScaleState, with no device involved. |
| [`test_memory_plan.cpp`](test_memory_plan.cpp) | That the memory planner never shares a byte between two values live together. |
| [`test_module.cpp`](test_module.cpp) | The parameter walk over layers, containers and a layer enumerated at run time. |
| [`test_objective.cpp`](test_objective.cpp) | The objective seam: the leaves it declares, the loss it emits and the resulting IR. |
| [`test_ops.cpp`](test_ops.cpp) | That the operation table is complete and that the reverse lookup resolves names. |
| [`test_precision.cpp`](test_precision.cpp) | The mixed-precision policy and where the cast insertion pass places its casts. |
| [`test_rematerialise.cpp`](test_rematerialise.cpp) | The structure of the graph that the IM2COL rematerialisation pass rewrites. |
| [`test_rule_table.cpp`](test_rule_table.cpp) | That every domain's backward rules reach the combined table exactly once. |
| [`test_shape.cpp`](test_shape.cpp) | Shape's extents, indexing, validation, equality and hash. |
| [`test_spatial_shapes.cpp`](test_spatial_shapes.cpp) | The shapes the builder infers for spatial operations and the layouts it refuses. |
| [`test_version.cpp`](test_version.cpp) | That the runtime version agrees with the compile-time macros. |

## test_backward.cpp

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

### EveryImplementedOperationHasARule

`has_backward_rule(BATCHNORM_UPDATE_STATS)` is false, and that is decision CV-7, not an
omission. A running mean is a buffer the training step writes beside the graph. No gradient
flows through it.

`has_backward_rule` is false for LEAF and ZERO because both produce values from nothing. There
is nothing upstream to hand a gradient to, so there is no rule to write.

`has_backward_rule(CAST)` is false because of the order of the pipeline. Casts are inserted
after differentiation (`include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

### EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `src/autograd/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/functional/integration/test_autograd_numeric.cpp` and all run in
FLOAT64, for the reason `include/SushiAI/autograd/gradcheck.hpp` derives.

The LOG_SOFTMAX entry names a message test. LOG_SOFTMAX's rule exists only to produce a good
error, so there is no gradient to verify and the message test is its proof.

### TwoLayerMlpProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

### TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another MATMUL. It is skipped, not
computed and discarded.

### FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered ACCUMULATE form.

### ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

### ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

### AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

### LogSoftmaxOutsideTheFusedPairSaysSoClearly

A RELU consumes the log-softmax result, so the cross-entropy fusion does not apply and the
standalone rule has to refuse.

## test_checkpoint_format.cpp

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `include/SushiAI/io/checkpoint_format.hpp` out of
`include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

### Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

### DistinguishesTwoShapesWithTheSameElementCount

This case covers the defect the format exists to close: an element count alone lets a [2, 3]
weight load into a [3, 2] one. The shape is recorded, so the two records are not
interchangeable.

## test_config.cpp

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`tests/common/test_throws.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document, and the layer registry it selects through
- the train and eval documents

### KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

### AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

### AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

### WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

### AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

### ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

### ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

### AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

### Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

### LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

### AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

## test_diagnostics.cpp

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`tests/common/test_throws.hpp` defines SA_EXPECT_ERROR.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### ALossThatNoRuleCanSeedIsRejectedRatherThanReturningNothing

A 1-element MATMUL result passes the "is it a scalar" check. Every rule then declines to act,
because the root's own gradient is implicit and only NLL_LOSS knows to start from it. Before
this diagnostic existed the caller got a full set of NO_VALUE gradients and no error. Then
`bind_parameters` reported a misleading "parameter receives no gradient", which blamed the model
and not the loss.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### LogSoftmaxOutsideTheFusedPairExplainsTheRestriction

Naming the way out matters more than naming the failure. The caller has to be told to route
through NLL_LOSS, and not merely that log-softmax said no.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.

## test_fusion.cpp

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
list. It no longer is. GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal became a condition: an INPUT-gradient activation may be folded only by an
entry that writes the tensor its gradient reads, and only if it writes the right one. That
condition has a compile-time half in `src/graph/fusion.cpp`, where an entry breaking either half
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

`activation_survives_fusion(RELU)` is the property the static_assert in `src/graph/fusion.cpp`
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

### ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

### AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

### AnIntermediateTheCallerReadsByIdIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
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
GEMM_BIAS_GELU with no spill would fail it. The fused node still writes the pre-activation as
its second output, and it is the same value, so `gelu_backward`'s operand needed no rewriting
and the backward rule needed no change.

`fused()[0].eliminated` holds only the MATMUL's result. A report that listed the spilled value
as eliminated would read like the report of the silently wrong version of this pass.

### ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the GELU's position, so a reader
of the pre-activation sitting between the bias_add and the GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

### ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

### TheGeluFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

### DisablingTheGeluFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::DISABLED`, is the only one.

### ARecoverableActivationNextToAFusedGemmIsFoldedNotReported

This case mirrors DisablingTheGeluFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

### ValueIdsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

### TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

## test_gradcheck.cpp

Tests the gradient checker against closed-form functions and a wrong gradient.

This file needs no device. The gradient checker is the instrument the integration tier measures
with, so this file calibrates it against closed-form functions whose right answer is known. One
of them is a deliberately wrong gradient, which proves the check can fail.

### RejectsAGradientThatIsWrongByATenthOfAPercent

This case is the reason for insisting on FLOAT64. A 1e-3 relative error is far above the noise
floor of about 1e-9 in FLOAT64. It would sit inside the floor of about 2.4e-5 that an fp32
difference quotient bottoms out at, and would pass unnoticed.

### AReluFamilyFunctionNeedsToBeSampledOffTheKink

`options.min_step` is large enough that the difference survives the cancellation of summing it
against the O(1) terms. A step of 1e-4 keeps the pushed coordinates at 1e-3, where a central
difference of x^2 is still exact to 1e-9.

The `unshifted` point straddles the kink. The quotient taken across it reports the average of
the two one-sided derivatives, which no correct rule can match.

## test_ir.cpp

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

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

### AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
`src/graph/lowering.cpp` reads slots with `get()` and passes the values straight to SushiBLAS.

## test_layer_stack.cpp

Tests that a LayerStack names and declares its parameters exactly as Sequential does.

This file needs no device. A LayerStack is host-lane, and the property this file pins, that it
names its parameters exactly as Sequential does, is a property of the walk and not of any
memory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

### NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

### KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

### TracingTwiceDoesNotDuplicateParameters

`Parameter::declare` is idempotent, and the run-time path must inherit that and not reintroduce
the duplicate-on-retrace bug at this layer. Both traces go into one graph, because a
parameter's id names a value in the graph that declared it. Retracing into a second graph would
reuse an index that means something else there.

## test_layout.cpp

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### EveryValueIsNhwcUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

### FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data.

### LayoutConvertIsAllowedToChangeIt

LAYOUT_CONVERT is the one exemption, and the reason the check runs per node and not as a single
sweep over every value.

### DeclareLikeInheritsTheSourceValuesLayout

Fourteen tasks still to come each emit a value and carry its layout forward by hand.
`declare_like` is the one place that does it, so a forgotten carry cannot happen.

### Crossed

Both rebuilding passes copy the value table entry by entry, so that copy is what the `Crossed`
graph tests, and not the pass's own subject. The chain is deliberately plain for the same
reason: a fusion candidate or a precision decision in it would only obscure which field went
missing.

### FusionCarriesTheLayoutTagAcrossItsRebuild

`fuse()` rewrites the node list and copies the value table. A field it forgets to copy is reset
to `add_value`'s default, which is NHWC. A dropped tag is therefore a wrong layout, not a
missing one, and every shape check downstream still agrees with it.

The cost of a dropped tag shows in `Lowering::record_layout_convert`, which decides whether to
relabel a side into [N,H,W,C] by reading these two tags. A genuine NCHW -> NHWC convert that
reads as NHWC -> NHWC applies no transpose and copies transposed data straight through.

### CastInsertionCarriesTheLayoutTagAcrossItsRebuild

Cast insertion is the other rebuilding pass and could make the same omission. Trainer runs both
passes unconditionally, so either one is enough to lose the tag on the default path.

### ARebuiltGraphStillProvesUniform

The consequence is worth stating separately. With every tag reset to one value, `finalise()`
passes for the wrong reason. A graph that is uniform because it was flattened cannot be told
from one that was uniform to begin with, so the proof has to be re-run against tags that
survived.

### TheThreeRulesThatEmitZerosCannotBeReachedWithAnNchwOperand

The test checks why the zeros exposure is latent and not live, and does not leave that to a
comment. It is the premise the `zeros_like()` change rests on, and it is a premise about other
emitters' guards, which is the kind that stops being true without anyone editing this file.

The two `EXPECT_THROW` lines on `slice_channels` and `channel_affine` cover the rank-4 sites,
`slice_channels_backward` and `channel_affine_backward`. CV-3 makes their forward emitters
refuse an NCHW operand outright, so their dy can never carry that tag.

The `reshape` to `rows` covers the rank-2 site, `sum_rows_backward`. Reshape carries an NCHW
tag into rank-2 land with no guard at all, so unlike the two rank-4 sites the emitters let this
chain be built.

The last `EXPECT_THROW(graph.finalise(), Error)` shows what blocks that chain: the far end.
`differentiate()` only seeds a loss produced by NLL_LOSS, whose one-hot operand is a leaf and so
always NHWC, and nothing but LAYOUT_CONVERT may cross back, which is rank-4 only. `finalise()`
rejects that node before differentiation ever runs, so the rule is never reached with an NCHW
gradient. The guard is the pre-existing proof in `finalise()`, not a check in any emitter, and
that is why it is worth a test and not a sentence.

### ZerosLikeCarriesTheTagAndPlainZerosCannot

The three backward rules now depend on this contract. ZERO reads nothing, so plain `zeros()` has
no operand to inherit from and its NHWC tag is a default, not an answer. That is harmless for
the ZERO node itself, which the per-node proof skips for having no inputs. The consumer that
reads those zeros beside a real operand is the node that would be seen crossing layouts.

### FinaliseStillPropagatesRequiresGrad

The rename must not quietly drop the behaviour the old name described. The assertion is the
same one `test_ir.cpp` made.

## test_loss_scaling.cpp

Tests the dynamic loss-scaling rule on LossScaleState, with no device involved.

This file tests the dynamic-scaling rule with no device in the measurement.

LossScaleState is split out of LossScaler so that this file can exist. "The scale doubles after
exactly N clean steps" is a property of the rule, not of a kernel. Asserting it through a
training run would make every one of these cases cost a compile and a replay while proving less.

The control matters as much as the case. A growth test that only ran N clean steps and saw a
doubling would pass against an implementation that doubled after every step, so each growth
assertion in this file is paired with the N-1 run that must not double.

### OneStepShortOfTheIntervalTheScaleDoesNotMove

This case is the control for TheScaleDoublesAfterExactlyTheGrowthInterval. Without it, "the
scale doubled after four clean steps" is equally true of an implementation that doubles after
one.

### AnOverflowRestartsTheRunOfCleanSteps

Three clean steps since the overflow is one short of the interval, so nothing has grown back.
The scale is still the backed-off one.

### BackoffStopsAtTheFloorButStillSkipsTheStep

At the floor the scale cannot go lower. The step is still garbage and must still be thrown
away, or an overflowed gradient reaches the weights the moment the backoff bottoms out.

### ANonDyadicConstantIsRefusedByName

Every constant is a power of two so that scaling and unscaling change only the exponent field.
A scale of 3 would work and would quietly make every gradient in the model carry a rounding
error, so it is refused.

### EveryReachableScaleHasAnExactReciprocal

The loss-scaling design is arranged around this property, and the test asserts it on the rule
itself. Whatever sequence of backoffs and growths the machine takes, the scale is a power of
two, so the reciprocal is exact and the round trip is lossless.

## test_memory_plan.cpp

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

ACCUMULATE's result is its accumulator's storage. Planning the two separately would be silently
wrong and not merely wasteful: the planner would think the accumulator died at the ACCUMULATE
node and hand its bytes to `sink`, which is produced by the node still reading them.

### EveryStartIsAWholeNumberOfItsOwnElements

Alignment is turned down to 1 so that the element-width floor is the only thing left doing the
work. A 20-byte float32 buffer at offset 0 leaves the next free byte at 20, which is not a whole
number of float64 elements. The float64 buffer therefore has to start at 24, or its
storage_offset would not be representable at all.

### TooManyNodesDeclinesOverlapRatherThanApproximatingIt

The refusal is the point. A planner that quietly fell back to overlapping linear intervals
would produce a plan that is wrong in the way TwoUnorderedBranchesNeverShareBytes describes, and
would report a saving for it.

### AGraphThatIsNotInTopologicalOrderThrows

GraphBuilder cannot produce this graph, because it always emits a producer before anything can
read it. The whole pass rests on the property all the same, so the test checks it. A closure
built over a back edge would silently miss ordering and the plan would overlap things it must
not.

### ThePlanIsDeterministicForTheSameGraph

The offsets are byte-identical across two builds. A plan that shuffled would move every tensor
and take the library's bit-for-bit replay guarantee with it.

### TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes

`FusedForm::GEMM_BIAS_GELU` writes two values from one node: its result and the pre-activation
its epilogue spills. They have the same shape, they are born at the same instant, and the
planner packs by liveness. "One died before the other was written" is the mistake available to
it, and taking it would have the GELU computed from its own output with nothing to show for it.

Nothing special prevents it, which is why it is worth pinning. `precedes()` asks whether every
use of the earlier buffer strictly reaches the later one's first write, and a node does not
reach itself. The planner therefore refuses both directions and the two values conflict. That is
an inference from an existing rule and not a rule of its own, so a future rewrite of
`precedes()` could take it away silently.

## test_module.cpp

Tests the parameter walk over layers, containers and a layer enumerated at run time.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds three groups: the walk over the library's own layers and containers, the
tutorial's custom layer, and the run-time arm of the walk.

### AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

### SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

### TracingTwiceDoesNotDuplicateParameters

The second trace emits nodes but declares no new parameters, so the value count grows by
exactly the two intermediates.

### The tutorial's custom layer

`MyOffset` is the custom layer of `docs/getting_started/TUTORIAL.md` §6, compiled.

A tutorial whose code sample does not compile is worse than no tutorial, and prose cannot be
made to fail a build. So the sample lives in this file as well, and the two must be kept
identical: if you edit one, edit the other.

TheTutorialsCustomLayerTracesAndIsWalkable and TheTutorialIsRightThatMulWouldNotBroadcast also
pin the two claims §6 makes about the sample: that bias_add is required and is not a stylistic
choice, and that the two ways of getting `fields()` wrong fail differently.

### TheTutorialsCustomLayerTracesAndIsWalkable

The test checks the role the tutorial gives the parameter, because the role decides the
initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.

### TheTutorialIsRightThatMulWouldNotBroadcast

§6 says bias_add is required and not merely preferred, because `mul` goes through
`require_same` and this builder has no implicit broadcast anywhere. That is the whole reason the
sample reads the way it does, so the test asserts it and does not leave it as a remark.

### The run-time arm of the walk

`WalkableStub` exercises the third arm of the walk: a layer whose parameters are enumerated at
run time.

The stub is deliberately not a LayerStack. This file tests the walk, and the walk's contract is
`walk_parameters(prefix, ParameterVisitor)`. Anything that satisfies it is enumerated through
it, whether or not it is one of the library's own types. Testing the walk against LayerStack
would tie its test to a type that arrived two commits later.

### AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

### TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
restates it so that the property is visible in the test that covers the walk, and so that it
names the detector and not the arm's consequences.

## test_objective.cpp

Tests the objective seam: the leaves it declares, the loss it emits and the resulting IR.

This file needs no device. Declaring leaves and emitting a loss are IR-to-IR, so the whole
objective seam is testable without an Engine. Only `score()` needs a tensor, and the integration
tier covers that through the trainer.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIr. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

### TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

### TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

### CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

## test_ops.cpp

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`include/SushiAI/core/ops.hpp`. These tests cover what that assert cannot: that the table is
complete, and that the reverse lookup behaves.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is IM2COL + MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReluDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

## test_precision.cpp

Tests the mixed-precision policy and where the cast insertion pass places its casts.

This file needs no device. The precision policy is a table of judgements and the cast pass is a
pure function of the IR, as the fusion pass and the memory planner are.

The cases pin placement, not survival. A mixed-precision graph that runs proves almost nothing:
a missing cast throws in SushiBLAS, so "it ran" only rules out the loud half of the failure
space. The quiet half, a cast in the wrong place, leaves a graph that trains and learns worse.
So every case is one of these:

- the whole-graph invariant (every node's operands agree with the type it computes in), asserted
  by walking the graph and not by spot checks,
- a control (the same graph under the fp32 policy must get zero casts, computed and not
  short-circuited), or
- one of the two places the master-weight arrangement can silently break: a parameter retyped in
  place, or a gradient that reaches its fp32 buffer without being widened.

The cases fall into five groups, in file order:

- the policy itself
- the control: the fp32 policy changes nothing
- the invariant
- master weights, and the direction that goes silently wrong
- what the pass promises about ids, reuse and repetition

### Traced

The `Traced` graph is built by hand and not through NN::Sequential, so that the test can name
every value it wants to assert about. It is also the smallest graph that still contains all four
interesting shapes: a parameter read by both passes, a reduction, a loss, and a gradient buffer.

### OnlyTheProductAndWhatAGemmFoldsIntoItNarrow

The `narrows` list states the rule set as a table, so that adding an operation to the narrowing
set is a visible edit in this test and not a silent behaviour change. Everything absent from it
stays in the graph's own precision, which is the safe answer for an operation nobody has thought
about yet.

The LEAKY_RELU entries are in `narrows` because LEAKY_RELU joined `Ops::ACTIVATIONS` with
`GradientSource::OUTPUT`. It narrows by the same substitution argument as RELU.

The `stays` list opens with the reductions and anything that sums many terms. fp16 carries about
11 bits of significand and SushiBLAS's own reduction header documents a 2048 ceiling for HALF,
so these are the operations that go quietly wrong.

DYNAMIC_SCALE is in `stays` because it carries the gradient loss scale, the one number in the
graph that exists to keep values out of fp16's subnormals. Narrowing the node that applies it
would overflow the very quantity it protects; see `include/SushiAI/optim/loss_scaler.hpp`.

GELU, GELU_BACKWARD and GEMM_BIAS_GELU, the one fused form that folds GELU, are in `stays`. The
form is a GEMM and rule 1 would otherwise take it. Narrowing it puts the tanh approximation's
`x*x*x` into fp16, which overflows at |x| ~ 40.3. Rule 5 in
`include/SushiAI/graph/precision.hpp` says so, and also explains why that leaves rules 2 and 5
consistent and not merely quiet.

The twenty spatial and CV identities are in `stays` because none of them is a GEMM epilogue or
an activation this policy folds. Each gets the same safe default as everything else nobody has
special-cased: it stays at the graph's own precision.

The closing `EXPECT_EQ` on the two list sizes holds because every identity is accounted for by
`narrows` or by `stays`. A new operation cannot join the enum without someone deciding which
list it belongs to.

### GeluStaysWideEvenThoughACatalogEntryNowFoldsIt

Rule 5 used to rest on two arguments, and one of them is gone. `FusedForm::GEMM_BIAS_GELU`
folds GELU by spilling the pre-activation, so "nothing folds it, therefore nothing forces its
precision" is no longer a reason for anything. The surviving argument is the arithmetic: the
tanh approximation computes `x*x*x`, which overflows fp16 at |x| about 40.3. A unit test cannot
assert that without a device.

The first three assertions check what a unit test can: that the conclusion did not quietly
follow the dead premise. The catalog can fold GELU and the policy still keeps it wide, which is
the pair of facts rule 5 in `include/SushiAI/graph/precision.hpp` now asserts.

`narrows_under_mixed_precision(GEMM_BIAS_GELU)` is false because the fused form has to go with
GELU. Otherwise the cast pass would narrow a node whose epilogue is the very expression rule 5
refuses to put in fp16.

### ChoosingIsIdempotentForEveryOperation

`include/SushiAI/graph/precision.hpp` states this contract, and it is what makes the invariant
checkable after the pass has run and not only while it runs. Without it,
`find_dtype_mismatches` would report a graph the pass had just finished making correct.

### AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType

The checks on the fp64 and fp16 graphs say that an fp64 graph keeps fp64 masters and still gets
fp16 products, and that an fp16 graph is already at the floor and is left alone, not "narrowed"
to something wider.

The `to_single` policy narrows to fp32. That is a legal thing to ask for, and it is what an fp64
model would use.

### TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Unit_CastInsertion cases measures nothing.

### EveryNodesOperandsAgreeWithTheTypeItComputesIn

This assertion is worth more than a dozen spot checks. It walks the whole graph and states the
rule, so it keeps holding as operations are added.

### TheInvariantSurvivesTheFusionPassThatRunsAfterIt

Fusion rewrites nodes and re-pairs operands, so it is where a correctly cast graph could stop
being one, and where a master weight could get swallowed.
`include/SushiAI/graph/cast_insertion.hpp` argues the ordering, and this test is the assertion
behind the argument.

### ADeliberatelyBrokenGraphIsReportedRatherThanAccepted

`find_dtype_mismatches` is the test file's own instrument, so it has to be shown to fail on a
graph that is wrong. Otherwise every "no mismatches" in the earlier cases could come from a
checker that never looks.

`mismatches.size()` is 3 because the graph is an uncast fp32 matmul under a mixed policy: two
operands and one result, all three float32 where the policy says half.

### AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy

The fp16 value the GEMM reads is a different value from the parameter. A cast produces it, and
it lives with the step's scratch and not for the whole run.

### AGradientIsWidenedBeforeItReachesItsBuffer

The test covers the backward direction. A matmul computing in fp16 produces an fp16 gradient,
and if that reached the fp32 gradient buffer uncast, SushiBLAS would throw. If the buffer had
been retyped to fp16 instead, nothing would throw and the optimizer would accumulate in fp16,
which is the failure master weights exist to prevent.

The loop over `gradients[0]` and `gradients[2]` covers the weights only. A backward MATMUL,
which computes low, produces the gradients of w1 and w2. SUM_ROWS, which does not, produces
those of b1 and b2, so only the weights need widening.

The loop over `gradients[1]` and `gradients[3]` covers the biases. Their gradients come out of
a reduction that stayed wide, so nothing was widened into them and nothing needed to be.

### AccumulatingGradientContributionsHappensWide

A parameter read twice gets two gradient contributions and an ACCUMULATE to sum them. Two things
have to hold at once. The sum runs in the wide type, because it is a sum. Its result and its
first operand agree, because the lowering makes the result alias the accumulator's buffer, so a
disagreement there is a wrong-width write and not merely a wrong number.

### AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot

This case covers the half of the pinning rule that a lifetime class does not. The logits are an
ordinary activation. The training loop reads them back by id for accuracy, and a caller who
forgets to say so gets an fp16 buffer where their host-side argmax expects floats.

### AValueReadTwiceIsConvertedOnce

w2 is read by the forward matmul and again by the backward one. Two conversions would be two
kernels and two buffers for one number, and the reuse is safe because the IR is
single-assignment.

### AGraphThatIsNotInTopologicalOrderIsRefused

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first.

### Pooled

MAXPOOL2D is the first operation in this IR whose results are not all one element type: it
writes the pooled tensor and, beside it, the INT64 argmax its backward reads. The cast pass
derives one precision per node from `outputs[0]`, which makes the argmax look like a value in
the wrong type and not like a value with no precision to choose.

`x` is a parameter and not an input for the reason
`tests/functional/regression/test_argmax_lifetime.cpp` gives: MAXPOOL2D_BACKWARD is only emitted
when something upstream of the pool needs a gradient.

### AnIndexBufferHasNoPrecisionToChoose

The pass forced every operand and every result of a node to one dtype, derived from the node's
first result. For MAXPOOL2D that is FLOAT32 from the pooled tensor, so the INT64 argmax read as
a mismatch and the pass tried to convert it. `GraphBuilder::cast` refuses anything but
real-float to real-float by design, so the whole build threw.

The defect does not depend on the policy. `wanted` comes from `outputs[0]`, not from the
policy's `choose()`, so the uniform policy reaches it too.

### APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer

This is the loud half of the same defect. An unpinned argmax is retyped in place and the
mismatch disappears silently. A pinned one keeps its id and its INT64 buffer, so the pass
reaches for the widening cast, and CAST is real-float to real-float by design.

## test_rematerialise.cpp

Tests the structure of the graph that the IM2COL rematerialisation pass rewrites.

This file needs no device. The IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Integration_Rematerialise in `tests/functional/integration/test_rematerialise.cpp`
does that by running the same graph twice.

The brief's three cases are all in this file, with two deviations that the library forced:

1. `Autograd::differentiate()` refuses any root that is not a scalar produced by NLL_LOSS
   (`src/autograd/differentiator.cpp` explains why every other root would silently return empty
   gradients). The brief's graphs stopped at a MATMUL, so each is carried on to a loss. The
   shapes, geometry and expectations are the brief's.
2. A "backward consumer" cannot be identified by node index, because the pass runs when the
   backward nodes are already in the graph and no marker separates the phases. The rule the pass
   applies is "every consumer after the first", which is the same set for every graph
   differentiation produces and is well-defined for one it did not produce.

### TheColumnBufferStopsBeingAnActivation

This case checks the number CV-10 is about. ResNet-18's first block at batch 128 holds
`128*32*32*64*9*4` bytes, about 302 MB, in this one buffer, and there are twenty convolutions.
Cloning the gather next to its backward consumer drops the lifetime to SCRATCH, so one buffer is
live at a time and not twenty.

### TheCloneComputesTheSameThingAsTheOriginal

The operand and attribute checks guard against a clone that read a different operand or carried
different attributes. Such a clone would produce a plausible column matrix and a wrong weight
gradient. Both are checked because the geometry lives in the attributes and is easy to drop when
copying a node.

The check on `copy` expects the same description and the same layout. The layout is a
graph-wide invariant, and a clone that dropped it would fail `finalise()` on a graph that is
numerically correct.

### TheCloneIsInsertedAheadOfEveryConsumerItServes

The failure this case catches is not a wrong number. It is a graph whose insertion order stopped
being a topological order, with a recomputation appended after the GEMM that reads it.
`MemoryPlan::build()` rests on that order and says so by name, and a lowering that got past it
would record a GEMM against a buffer nothing had written yet.

### AnIm2colWithNoBackwardConsumerIsLeftAlone

This is the inference case. Nothing was differentiated, so there is no second consumer and
cloning would be a free slowdown.

### AForwardOnlyConvolutionIsLeftAlone

This is the other half of the inference case, and the one that would break if the pass cloned
for every consumer and not for every consumer after the first. The column matrix has exactly one
reader, the GEMM.

### RunningItTwiceAddsNothingTheSecondTime

Once the rewiring is done the clone has one consumer of its own, so the pass has no reason to
rematerialise its own output. If it did, the pass would not terminate on a graph with three
consumers.

## test_rule_table.cpp

Tests that every domain's backward rules reach the combined table exactly once.

### EveryDomainsRulesReachTheCombinedTable

`include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

### NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

### TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

## test_shape.cpp

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
RuntimeContext, which is what keeps the unit tier fast.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

## test_spatial_shapes.cpp

Tests the shapes the builder infers for spatial operations and the layouts it refuses.

### Im2ColProducesTheColumnMatrixAGemmExpects

[N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5 windows per image. Each row is
one window's kernel_h * kernel_w * C taps with channels innermost, so a sub-group reads a
contiguous run (CV-3).

### Im2ColHonoursStrideAndDilation

A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows, and stride 2 over the same
input yields 2. Both numbers are wrong under the naive (H - K)/S + 1 that forgets dilation.

### Im2ColRejectsAWindowLargerThanThePaddedInput

The oversized window produces a zero extent, which Shape refuses anyway. The message a caller
gets should still name the geometry, not the Shape invariant it happened to trip.

### Col2ImReturnsToTheImageItWasAskedFor

col2im cannot derive its output extents. ConvGeometry carries pad_top and pad_left but no
pad_bottom or pad_right, so an asymmetric padding would be silently mis-shaped. The caller
states the image shape instead. It always has it, because it held the value im2col consumed.

### Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward

A 3x3 stride-1 window with padding only on the bottom and right of a 13x13 input still yields
13x13 windows, which is not what the symmetric formula (13-3)/1+1 = 11 would say. The forward
pass (`im2col_with_extents`) states 13x13 outright. The backward rule must be able to state the
same numbers back, because it is the one holding them, and must not re-derive them from
ConvGeometry alone.

The geometry leaves pad_top and pad_left at 0. Padding was applied on the bottom and right
only, which ConvGeometry cannot express and does not need to.

### Col2ImDerivingStillRejectsThatSameAsymmetricPair

This case is the point of having `col2im_with_extents` as a twin. The deriving col2im assumes
symmetric padding and is not a substitute for it. Feeding it the exact cols/image pair that
`col2im_with_extents` accepted in Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward must
still throw, which proves the twin is needed.

### Im2ColRejectsAnNchwInput

im2col reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it would read channels
as height and produce a plausible but silently wrong shape, where it should throw.

### MaxPoolProducesAValueAndItsArgmaxTogether

The indices are a second result of the node, not a second output tensor the caller allocates.
The backward rule needs exactly the indices this forward produced, and a pair returned together
cannot be mismatched the way two separate calls could.

### MaxPoolHandlesYoloTinysAsymmetricThirteenToThirteen

YOLOv3-tiny's last pool is 2x2 stride 1 over 13x13, padded only on the bottom and right, staying
13x13. ConvGeometry cannot express that padding, so the caller states the extents. This case
proves the explicit overload is needed.

### AvgPoolProducesOnlyAValue

avgpool2d records no indices: its backward divides by a constant window area and needs nothing
the forward recorded (CV-6).

### MaxPoolRejectsAnNchwInput

maxpool2d reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it must throw and
must not pool the channel axis.

### SpatialMeanReducesToNAndC

The result is the [N,C] that the numeric Probe's loss needs, which is why every convolutional
gradient test in this plan ends with this call.

### BatchNorm2dKeepsItsShapeAndDeclaresTwoPerChannelStatistics

A non-positive epsilon is a division by the standard deviation itself, which is exactly zero
for a constant channel.

### BatchNorm2dRejectsAnNchwInput

The layout check runs before any extent is read, so the operand that is wrong is the one the
message names.

### ReshapeRefusesToChangeTheElementCount

A reshape has one failure mode. It moves no data, so a mismatched count is not a resize; it is
a read past the end.

## test_version.cpp

Tests that the runtime version agrees with the compile-time macros.
