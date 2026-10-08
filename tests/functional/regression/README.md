# Regression tests

The functional regression tier holds the tests for guarantees that fail without a symptom: the run
still converges and still reports success. This document carries the reasoning the comments in
these files cite. Each section is named after one test file, and within it a heading names the
test or helper its paragraphs belong to.

## Shared conventions

### The SaTest alias

Inside a TEST_F body the fixture's base class injects the name `Test`, so `Test` alone resolves to
`::testing::Test`. Each file therefore declares `namespace SaTest = SushiAI::Test`, which keeps the
helpers in `SushiAI::Test` reachable from a test body.

## test_argmax_lifetime.cpp

Suites: `Regression_ArgmaxLifetime`.

K_MAXPOOL2D's argmax buffer has to survive from the forward node until K_MAXPOOL2D_BACKWARD reads it,
and no test pinned that before this file.

The gradchecks cover pooling, but they cannot see this defect. If the planner handed the argmax's
bytes to a later activation, the backward would read whatever that activation wrote and route every
gradient to a plausible wrong pixel. The gradchecks would keep passing until the planner changed,
and then fail somewhere else with no mention of pooling.

The test therefore asserts on the plan and not on a number. Reuse in a MemoryPlan is byte overlap
and nothing else: two values share storage exactly when their intervals in one arena intersect. An
argmax whose interval intersects no other buffer's owns its bytes for the whole step, and that is
what "spans forward to backward" means. The check reads the plan's own arena and placement tables.
It does not read a lifetime enum, because the enum is the input to the planner's decision and the
test is about the output.

### ThePlannerKeepsMaxPoolsIndicesAliveUntilItsBackward

The graph is a pool feeding a chain long enough that the planner has real candidates to reuse the
argmax's bytes for. On a two-node graph the overlap assertion would hold because there was nothing
to overlap with, which proves nothing; the non-vacuity check at the end of the test is what gives
the assertion its meaning.

`x` is declared as a parameter and not as an input. The pool's backward is only emitted when
something upstream of it requires a gradient, and an input does not. Differentiating with respect
to `x` is what puts K_MAXPOOL2D_BACKWARD in the graph.

Between the pool and its backward the test builds a spatial mean down to [2,1,1,4], a reshape to a
matrix and a small model, so several activations are born and die while the argmax waits to be
read.

`backward_reads_the_indices` checks the premise: differentiation emitted a consumer for the
indices. Without that consumer the rest of the test is a statement about a value nobody reads.

The test asserts `plan.is_packed(pooled.indices)` before it looks for overlaps. `overlaps()`
answers false for an unpacked placement, so an argmax that had been given its own allocation would
satisfy every later assertion for a reason this test does not document.

The `overlapping_neighbours` loop is the assertion the file exists for: every other buffer in the
plan must miss the argmax entirely. Aliased names for the same bytes (K_ACCUMULATE) are not reuse, so
the loop compares by owning buffer and not by value id.

`the_planner_reused_anything` is the non-vacuity check. It requires that the planner overlapped
some pair of buffers in this graph, so the missing overlap on the argmax is a decision the planner
made and not an opportunity it never had.

## test_compile_once.cpp

Suites: `Regression_CompileOnce`.

This file was the regression tier's first. It pins the two claims in this library whose failure is
completely silent.

"Trace once, differentiate once, compile once, replay for ever" is the premise the design is
arranged around. Both halves of it fail without a symptom:

- A trainer that recompiles per step still converges. It pays graph analysis every step, and
  nothing in the output says so.
- A replay that never dispatched still reports complete. CompiledStep exists because calling
  rearm() before capture_base_state() leaves the completion sentinel armed at in-degree zero, so
  run() returns having done nothing. The loss stops moving, which reads like a bad learning rate.

Nothing upstream can catch either. sushiruntime has no test that replays a raw CompiledGraph, and
sushiblas cannot see a training step at all. So both claims are pinned here, in the shape the
trainer and `sa demo mlp` use.

Each test in this file spells out its own setup and shares none of it through a fixture struct.
Trainer holds raw pointers into the pool and the optimizer, so the construction order (bind, then
initialise, then optimizer, then trainer) is part of the contract, and a test that spells the order
out fails loudly if that order ever stops working. The setup matches
functional/integration/test_training.cpp line for line for the same reason.

### ManyFitCallsCompileTheStepExactlyOnce

`sa demo mlp` drives one fit() per epoch so it can print a per-epoch trajectory. That call pattern
differs from a single fit() with epochs=N. The integration test calls fit() once, so it could not
see a recompile hidden inside fit().

### EveryReplayDispatchesRatherThanReportingCompleteImmediately

This test looks at the CompiledStep defect from the outside. An unarmed plan's run() returns
successfully having dispatched nothing, so the parameters never move and the loss goes flat. The
defect shows from the second replay onward, because the first replay runs off the state that
compile captured.

### TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline

This test catches the rearm/capture inversion at the first run, where
EveryReplayDispatchesRatherThanReportingCompleteImmediately catches it only at the second replay.

That other test reads the defect off a loss that stops changing, so it needs two replays to see
it. But the inversion breaks the first run too. With an empty base state, rearm() writes the
completion sentinel's in-degree as zero, the sentinel is ready before anything is dispatched, and
execute() returns success having run no kernel. The only thing that distinguishes that from a step
that ran is the contents of the output buffer.

So the test first poisons the buffer with a value the graph cannot produce, and it asserts that the
poison is gone. A run that dispatched nothing leaves the poison where it was.

The test was verified by deleting the capture_base_state() call from the compile path in
`src/graph/compiled_step.cpp`. It then
fails on the first assertion and reports the poison still in place.

The replay loop repeats the check three times. Each pass poisons the output again, changes the
input, and requires the output to track the new input and not the old answer or the poison.

### CompileCountSurvivesEverythingTheLoopDoesBetweenSteps

ManyFitCallsCompileTheStepExactlyOnce asserts compile-once for the plain path. This test pins that
none of the things a real loop does between steps reaches back into the plan: rewriting the batch
buffers, interleaving an optimizer that records and flushes its own graph every step, stopping and
resuming, and changing a hyperparameter mid-run.

The test builds two datasets of the same shape with different contents. The compiled plan reads its
inputs by address, so swapping the data a step trains on must be a copy into those buffers and
nothing more.

The `round` loop stops and resumes: fit() returns and is entered again, which is the pattern
`sa demo mlp` uses to print a per-epoch trajectory.

The loop also halves the learning rate each round. A learning-rate schedule changes a host-side
scalar the optimizer reads per step. If that scalar ever became a compiled attribute it would force
a recompile; ir.hpp's warning on Attributes says the same.

## test_determinism.cpp

Suites: `Regression_Determinism`.

This file tests bit-for-bit reproducibility from end to end.

Reproducibility is the guarantee the stack is arranged around, and no test proved it before this
file. SushiRuntime promises sushiruntime_deterministic_fp. autograd/tape.cpp sums a fan-out through
an ordered K_ACCUMULATE chain and not through float atomics, and its own header says an atomic sum
"would break that quietly, showing up as a training run that cannot be reproduced rather than as a
test failure". A claim whose failure mode is "not a test failure" needs a test, and the test has to
compare bits. EXPECT_NEAR passes on a run that reorders a reduction, and reordering a reduction is
the change the claim forbids.

The file sits in the regression tier because every failure here is silent. A model that trains to
a slightly different place each run still converges and still reports good accuracy. The defect
shows on the day someone needs to reproduce a result.

Every test in this file builds a fresh Engine per run, and that is part of the contract. SushiBLAS's
set_seed() leaves the engine's monotonic RNG offset counter alone, so two generators built on one
engine draw different streams from one seed. dataset.hpp documents this.

### train_once

Everything the run depends on is either fixed in this helper or seeded: the model shape, the
initialisation seed (InitPolicy's default), the data seed and the optimizer. Two calls therefore
differ in nothing, so a difference in their output is a finding.

### expect_identical

The loss is read out of device memory as a float. Two identical runs must produce the same float,
so `==` is the right comparison. EXPECT_NEAR would admit that the two runs might differ.

### TwoIdenticalTrainingRunsAgreeByteForByte

The run is two epochs with momentum. That is enough steps for a reordered reduction to accumulate a
visible difference. With momentum, step N depends on every step before it, so a single divergent
bit cannot cancel out.

### TwoIdenticalAdamWRunsAgreeByteForByte

AdamW divides by the square root of an accumulated second moment. After that division a one-ULP
difference in the gradient is no longer one ULP.

### InitialisationFromOneSeedIsBitwiseIdentical

The starting point has to be identical before anything downstream can be. Initialisation is also
the only part of the run that goes through the device RNG and not through arithmetic.

### TheSyntheticDatasetIsBitwiseIdentical

The existing dataset coverage compares through EXPECT_DOUBLE_EQ, which tolerates four ULP and
cannot see a sign difference on a zero. A dataset is either reproducible or it is not, so this test
compares the bytes.

### ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable

This is the narrowest form of the claim, with the optimizer and the RNG removed: the same compiled
plan and the same buffers, run twice. Anything the runtime does non-deterministically shows up here
with nothing else to blame it on. A work-stealing reduction, an atomic accumulation and a
scheduler-dependent order are the candidates.

The optimizer's learning rate is zero. That keeps the parameters fixed, so every replay sees the
state the one before it saw.

## test_fused_optimizer.cpp

Suites: `Regression_FusedOptimizer`, `Regression_FusedOptimizerDeathTest`.

This file is the gate on collapsing eleven launches per parameter into one.

Fusing an optimizer update is a pure performance change, and it touches the only arithmetic in the
library that nothing downstream validates. Every failure mode it has is silent: a transposed scalar
index, an epsilon folded inside the square root, a momentum buffer read a step late, a first-step
branch that stopped being special. Each of those still trains, still converges and still reports a
healthy loss. So the fused kernel is not tested against a tolerance or a fixture. It is tested
against the separate-op sequence it replaced, byte for byte, from identical state.

The tests make five claims, in file order:

1. One kernel computes what eleven did: AdamWsFusedKernelMatchesTheSeparateOperationsExactly,
   AdamWWithoutDecayTakesTheOtherBranchAndStillMatches and
   SGDsFusedKernelMatchesTheSeparateOperationsExactly.
2. Folding the update into the plan changes no number:
   TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt and
   TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo.
3. The step compiles once, with the update inside it: TheStepCompilesOnceWithTheOptimizerInsideIt.
4. The launch count: TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve.
5. An update that cannot be replayed is refused and not downgraded:
   AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored,
   SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused and
   TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored.

The helpers from `K_BATCH` to `expect_runs_agree` belong to claims 2 and 3, which train a whole model
through the Trainer.

### Why byte for byte is the right bar

Each element's update is independent and the operation sequence is unchanged, so the only thing
that could move a bit is a difference in rounding. The one plausible source of that is a compiler
contracting `a * b + c` into an FMA inside one kernel where it could not across two. The build
rules it out: the whole stack compiles with `-ffp-contract=off` through SushiRuntime's
`sushiruntime_deterministic_fp` contract. Every expression in the fused functors is also written in
the same operand order as the kernel it replaces (`x *= alpha` is `x * alpha`; `y += alpha * x` is
`y + (alpha * x)`). Both paths read their scalars out of the same float32 device buffer, so neither
can pick up a differently rounded constant.

If a comparison in this file ever fails, the response is to find which expression moved. Loosening
the comparison is not a response.

### The second half: folding the update into the compiled plan

Fusing makes the update replay-invariant, and replay-invariance is what lets the update be recorded
into the same TaskGraph as the forward and backward passes. That is a separate change with its own
silent failure mode. A dependency edge the runtime does not see would let the update race the
backward pass that produces its gradients, and a race that usually resolves the right way looks
like training. So the folded and unfolded placements are compared byte for byte too, and
compile_count() is pinned at one with the optimizer inside the plan.

### K_ELEMENTS

`K_ELEMENTS` is 133, which is not a multiple of the 128-bit vector width. Both the wide pass and the
scalar tail of Launch::zip_inplace therefore run on every call.

### gradient_for

The first four entries of the gradient are placed by hand and not sampled: an exact zero, a
negative zero, a denormal-adjacent tiny value and a large one. Zero is where AdamW's
`0 / (sqrt(0) + eps)` lives. Negative zero distinguishes `b = staged` from `b = 0*b + 1*staged`:
the second form turns -0.0 into +0.0, and no tolerance would see that. The two extremes exercise
the square root and the division away from unity.

### drive

The helper uses no graph, no trainer and no device RNG. It has one parameter, one gradient
rewritten from the host between steps, and the optimizer's own step(). The update is then the only
thing in the measurement.

### train

The helper builds a fresh Engine, because reproducibility is a property of one engine. dataset.hpp
explains the RNG offset counter behind that.

### AdamWWithoutDecayTakesTheOtherBranchAndStillMatches

With `weight_decay = 0` the decay factor is exactly 1, and both paths must skip the multiplication
and not multiply by 1. `theta * 1.0f` happens to be bit-identical here, but the skip is a separate
branch in each path and this test is what covers it.

### SGDsFusedKernelMatchesTheSeparateOperationsExactly

The test runs every branch of the SGD recipe, because each one is a separate line of the fused
functor and a separate chance to have written it differently.

### TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt

If the two runs diverge, the update is not ordered against the graph the way the separate flush
ordered it. The runtime is then missing a dependency edge, and the fix is that edge, never a
tolerance.

### TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo

This is the end-to-end version of AdamWsFusedKernelMatchesTheSeparateOperationsExactly. That test
drives one optimizer over a synthetic gradient. This one runs the whole pipeline, with gradients
the backward pass produced and a trajectory long enough for a one-ULP difference in an early
denominator to become a visible one.

### TheStepCompilesOnceWithTheOptimizerInsideIt

The model has four parameters, so four fused updates joined the plan. The test states the count as
a difference and not as an absolute, so a change to the graph itself does not have to be recorded
again here.

### TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve

This test counts the headline claim and does not time it. A wall clock says different things on a
loaded machine, but the number of tasks issued is the same everywhere, and it is the quantity the
change altered.

The separate AdamW update with decay issues twelve tasks: the decay scal, two for m, three for v,
four for the denominator, and two for the step. Without decay it issues eleven, which is the figure
the SushiBLAS seam's own documentation quotes.

The counting lambda runs the phases step() would run and reads the count out of the middle one. The
flush after it is required: the tasks are in the engine's builder and something has to drain them.

SGD's first step is its cheapest, because `b_1 = g_1` is one copy where later steps need a scal and
an axpy. Four is therefore the floor for SGD and not its typical cost.

### AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored

The separate spelling bakes the current step's bias corrections into its recorded tasks. Folded
into the plan, it would replay step one's corrections for ever and still converge. That is why the
request has to be an error and not a quiet fallback.

### SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused

Before the update joined the plan, `trainer.run_step()` called `optimizer.step()`, and a caller who
also called it got a visible double update. With the update inside the replay and invisible from
the outside, the same mistake would train at twice the learning rate and produce a run that looks
fine and is wrong.

### TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored

The momentum buffer is allocated at construction or not at all, so the fused kernel's operand count
is fixed there too. Before this check existed, the separate path dereferenced a null tensor and the
fused path would have trained without momentum.

## test_fusion_numerics.cpp

Suites: `Regression_Fusion`.

This file is the gate on collapsing the forward and backward graph into fewer kernels.

Fusion is a pure performance change over the only arithmetic in the library that nothing downstream
validates, and every way it can be wrong is silent:

- An epilogue that adds the bias at the wrong coordinate still produces a loss that falls, because
  the model learns a different bias.
- An activation folded into a GEMM whose gradient needed the pre-activation trains towards the
  wrong place and reports nothing.
- A fused kernel reading memory the dependency tracker cannot see races its own producer, and a
  race that usually resolves the right way looks like training.

So the fused graph is not tested against a tolerance. It is tested against the unfused graph it
replaced, byte for byte, from identical state.

The tests fall into four groups, in file order. The first claims that fusing changes no number:
AReluMlpTrainsIdenticallyFusedAndUnfused, ATanhMlpTrainsIdenticallyFusedAndUnfused and
ASigmoidMlpTrainsIdenticallyFusedAndUnfused. The second is the five GELU tests, from
AGeluMlpTrainsIdenticallyFusedAndUnfused to DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt,
which the section on the spilled activation describes. The third counts what fusion cost:
TheFusedStepIssuesTenFewerKernelsPerStep, AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias,
WhateverReadsTheSpillIsOrderedBehindTheGemmThatFillsIt and
DisablingOneCatalogEntryLeavesTheRestAlone. The fourth covers the dynamic selector, from
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove to EveryChoiceCarriesAReasonSomebodyCanRead.

### Why byte for byte is the right bar

The argument is the one test_fused_optimizer.cpp makes for the optimizer. Every element is computed
by the same operations in the same order. The one plausible source of a moved bit is a compiler
contracting `a * b + c` into an FMA inside one kernel where it could not across two, and
`-ffp-contract=off` rules that out; the whole stack inherits the flag through SushiRuntime's
`sushiruntime_deterministic_fp` contract. Each fused functor is written in the operand order of the
calls it replaces, and a constant that reached a kernel through a `float` parameter still does.

If a comparison fails, the response is to find which expression moved, not to widen the comparison.

### The second half: the activation that needs a spill to be fused

GELU's gradient differentiates at its input, so the epilogue that folds GELU has to write that
input to an auxiliary buffer on its way past. cuBLASLt calls this GELU_AUX_BIAS; here it is
FusedForm::GemmBiasGelu. Every way of getting it wrong is silent, and worse than the other
activations: a forward pass that matches perfectly while the gradients are taken at the wrong point
still converges, to a different place.

A forward comparison is therefore not the test. The test is that a GELU model trained end to end
reaches byte-identical parameters after 24 optimizer steps. That is a statement about the
gradients, since nothing else moves a weight. It is taken three ways, against three different
fillings of the same buffer:

- AGeluMlpTrainsIdenticallyFusedAndUnfused compares against the graph with no fusion pass at all.
- DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt compares against the same graph with only
  this catalog row switched off, so the difference is the entry and nothing else.
- ASelectorRefusalStillFillsTheSpillAndMovesNoBit compares against a selector that declines the
  fused arm, so the aux buffer is filled by a separate copy in the lowering and not from inside
  the epilogue.

TheSpillHoldsThePreActivationItself then reads the buffer directly and compares it with a
host-computed pre-activation, because three spellings can agree while all three hold the wrong
tensor.

### train

The helper builds a fresh Engine, because reproducibility is a property of one engine; dataset.hpp
explains the RNG offset counter behind that. It trains with AdamW and not plain SGD because AdamW's
division by an accumulated square root is where a one-ULP difference stops being one ULP.

### pretend_device

The one device on the test machine is an OpenCL CPU with no dedicated scratchpad, and it supports
doubles. A selector rule that can only be exercised on hardware nobody owns is a rule nobody has
run, so the declining branches are driven from a described device.

### AReluMlpTrainsIdenticallyFusedAndUnfused

This is the demo's own model. Both linear layers fuse, the first with its ReLU and the second
without one, and so does the cross-entropy gradient's elementwise tail. One run therefore covers
three separate collapses that reach the same arithmetic.

### ATanhMlpTrainsIdenticallyFusedAndUnfused

tanh's gradient reads its own output, so the pre-activation is dead and the whole layer folds. The
tanh epilogue is a different template instantiation from ReLU's, which is why it has its own
comparison.

### AGeluMlpTrainsIdenticallyFusedAndUnfused

This test runs the GELU hazard end to end, and its claim is about the gradients and not the forward
pass. A K_GEMM_BIAS_GELU that computed the right activation and spilled nothing would match the
unfused forward exactly and still differentiate at the wrong tensor. Nothing would throw, no value
would go non-finite, and the loss would still fall. What such a kernel could not do is land the
parameters on the same bits after 24 AdamW steps, because a weight only moves by its gradient.

The fused graph holds two fused linear layers, the first folding its GELU and the second with no
activation to fold, and the fused cross-entropy gradient tail.

The test also asserts that the loss fell. Without that assertion, "identical" could describe two
runs broken in the same way.

### TheGeluFoldIsReportedAsAFoldAndNotAsARefusal

The test asserts that the spilled value is the one the backward rule reads. It checks this on the
IR and not through a loss, because "the gradient is taken at the right tensor" has no numerical
symptom when it is wrong.

### TheSpillHoldsThePreActivationItself

The test reads the spill buffer directly and does not infer its contents from a downstream
consequence. Every other check on this catalog entry goes through gradients that agree, and two
spellings can agree while both hold the wrong tensor.

`pre` is left unobserved on purpose. The fold has to survive on the spill and not on a pin, or the
test would be pinning the wrong thing.

The reference the spill is compared against is computed in the test, on the host, from the inputs.
It is not read out of another tensor the same kernel wrote.

The test also asserts that the spill is not the post-activation, which is the value a fold without
a spill would leave for the backward pass. Where that check is made matters. gelu(x) tends to x for
large positive x, so the two tensors agree there to within a rounding error, and an assertion taken
where |x| is large and positive would be satisfied by either. On the negative side gelu tends to
zero, so a sufficiently negative element separates them by about |x|.

### ASelectorRefusalStillFillsTheSpillAndMovesNoBit

This test is the control for TheSpillHoldsThePreActivationItself. With the fused arm declined, the
aux buffer is filled by a separate copy in `record_fused_gemm` (`src/graph/lowering/fused.cpp`) and
not by the epilogue.
That is a different code path to the same buffer, and the gradients have to come out identical.
Deleting either write makes this comparison fail.

### DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt

This is the A/B comparison the catalog entry was written against: the same model with the catalog
row on and off. With the row off, the K_GELU is an ordinary IR node and the pre-activation is an
ordinary value. With it on, the pre-activation is a second output of a fused GEMM, written from
inside a device kernel. The parameters after 24 steps have to be the same bits, and they are the
same bits only if the gradients were.

Disabling the form costs two more kernels, as switching the ReLU form off does. The activation
becomes an out-of-place IR node again, which is a copy and a gelu. The spill costs nothing on the
unfused side, because the value the fused node spilled is the value K_BIAS_ADD writes anyway.

### TheFusedStepIssuesTenFewerKernelsPerStep

This test counts the headline claim and does not time it. A wall clock says different things on a
loaded machine, and the kernel count is the number the change controls.

The saving is ten for this model. The first layer's gemm+bias+relu replaces five calls (gemm; copy
and ger for the bias; copy and relu) with one. The second layer's gemm+bias replaces three with
one. The cross-entropy gradient's exp/sub/scale replaces five with one.

The saving has no cost beside it. Each fused GEMM used to need a host-side ordering node carrying
the dependency edges execute_gemm could not declare for a bias its functor dereferences. Two layers
meant two extra nodes. SushiBLAS's seam now takes the epilogue's own access set, so
`operations_per_step` is the whole plan again and the two counts cannot drift apart.

### AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias

The fused spelling carries a hazard the unfused one does not. The epilogue dereferences the bias
from inside the GEMM kernel, and the optimizer rewrites that same bias inside the same compiled
plan. SushiBLAS cannot see into the functor, so this library has to make the declaration:
`gemm_epilogue(f).reads(bias)` in `record_fused_gemm`.

The test asserts on `in_degree` and never on a loss. With no edge the two tasks may still come out
in the right order on any given run, so a falling loss proves nothing about ordering, and this
failure mode survives an otherwise green suite. The node count is asserted for a related reason:
the plan used to hold three nodes, the third being a host-side ordering node that ran nothing and
existed only to carry this edge.

The bias-writing task in the test stands in for the optimizer's update: a task in the same plan
that writes the bias the epilogue reads. It is recorded first, so an edge can only run from it to
the GEMM.

### WhateverReadsTheSpillIsOrderedBehindTheGemmThatFillsIt

This test covers the other half of the epilogue's access set, the half only the spilling arm has.
`.reads(bias)` orders the GEMM behind whoever writes the bias; `.writes(aux)` orders whoever reads
the spill behind the GEMM. Without the second declaration, the backward pass that differentiates at
the pre-activation races the kernel that produced it. A race that usually resolves the right way
cannot be told apart from training, which is why this test also asserts on `in_degree` and never
on a loss.

The test emits a reader of the pre-activation after the K_GELU, the way the differentiation pass
emits gelu_backward.

### DisablingOneCatalogEntryLeavesTheRestAlone

The per-entry switch serves two callers. TheFusedStepIssuesTenFewerKernelsPerStep needs it as its
A/B control, and a user who has measured a different result on their own model needs it as a way
out. Switching the three-op GEMM form off must fall back to the two-op form and not to nothing.

The catalog refusal costs two kernels more and not one, and the two ways of declining differ here.
Refused at the catalog, the ReLU stays an ordinary IR node. An IR activation is out of place, so it
lowers to a copy and a ReLU. Refused by the selector, as in
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove, the node is already fused and its output
tensor is the activation's own result, so the trailing ReLU runs in place and costs one. The
numbers are the same either way; the launch counts differ.

### ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove

The seam between the two stages makes one promise. The static pass decides what could be fused and
the selector decides what is fused, so a device that wants no fusion gets an unfused plan out of a
fused graph. The numbers stay the same because every spelling of a fused node is the same
arithmetic.

### AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory

Under ForceReference the epilogue is invoked on the host by Kernels::Reference::gemm,
so any pointer it carries must be host-readable there. SushiAI's own allocations are shared USM and
would be readable. A bound allocation need not be, and nothing in the selector can ask which kind
it got.

The zip-based form runs no host arm, so it is unaffected. A blanket refusal of every form in these
modes would be a different bug from the one being prevented.

### EveryChoiceCarriesAReasonSomebodyCanRead

A selector nobody can interrogate is a selector nobody will trust. The static pass carries a report
for the same reason: both of its answers produce a plan that trains.

## test_learning_curve.cpp

Suites: `Regression_LearningCurve`.

This file holds the test that catches "everything passes and the model stops training".

Every other test in the suite checks a property: this gradient matches a finite difference, that
step compiled once, this loss is finite. A model can satisfy all of them and still learn measurably
worse than it did last month. A gradient scaled by the wrong constant, a momentum buffer applied a
step late and an initialisation gain quietly changed all do this. Each leaves the loss going down
along a different path, and no property test has an opinion about the path.

So TheLossTrajectoryMatchesTheRecordedReference pins the trajectory itself against a reference
recorded from a real run of this code. The tolerance carries the design of the test:

- Too tight, and the test fails on every legitimate kernel change. Reassociating a gemm's inner
  sum moves an fp32 result by an ULP or two, and twelve steps of momentum amplify that. A tolerance
  below about 1e-3 would be noise.
- Too loose, and the test stops distinguishing training from not training. A gradient wrong by a
  constant factor, or a backward rule reading the forward output where it needed the input, moves
  this trajectory by tenths.

The head steps, 0 to 5, use the absolute bound K_LOSS_TOLERANCE, 0.02, which sits between those
two bounds. The tail steps, 6 to 15, use K_LOSS_RELATIVE_TOLERANCE, 2 per cent of each step's own
reference. From step 6 on the reference values are below 0.02, so an absolute bound there would
accept a loss ten times its reference: at step 15 the reference is 0.002215 and the bound is 0.02.
The relative bound pins each tail step to its own magnitude. The shape assertions in
TheTrajectoryHasTheShapeALearningModelHas (the exact ln(C) start, the per-epoch descent, the final
value) hold even if a future toolchain widens the noise floor.

Recording the reference again is legitimate only when the change that moved it is understood and
intended. If the trajectory moved and nobody knows why, the movement is the finding and the
tolerance stays where it is.

### TheTrajectoryHasTheShapeALearningModelHas

The shape assertions are stated independently of the recorded numbers, so they survive a legitimate
re-record. Together they say that the trajectory starts where a zeroed classifier must start, goes
down, and gets most of the way to the minimum.

The descent is not monotone step by step, because each step sees a different batch. Each epoch's
mean must beat the mean of the epoch before it.

## test_log_softmax_stability.cpp

Suites: `Regression_LogSoftmaxStability`.

This file tests a log-softmax that was stabilised by the wrong maximum.

The lowering used to shift the whole batch by the batch's global maximum, on the argument that
log-softmax is invariant to subtracting any single scalar from a row. The argument is sound in real
arithmetic and false in binary32. The exponential destroys the invariance, not the subtraction: a
row sitting a long way below the batch maximum has every `exp(x - global_max)` collapse into the
subnormals and then to zero, its row sum becomes exactly zero, `log(0)` is `-inf`, and the row's
log-probabilities come back `+inf`.

The figures in the table were measured on this project's OpenCL CPU device, in FLOAT32, on the
[2, 3] input `logits_with_gap` builds, whose true log-probabilities are about -0.41 to -2.41.

The quantity is the worst absolute deviation of row 1's log-probabilities from the answer the same
code gives at a gap of 0. It is the one figure measured directly on both sides of the fix, and the
assertion that produces it is the EXPECT_EQ in TheAnswerDoesNotDependOnTheGapAtAll:

| row-to-batch-max gap | global shift (before) | per-row shift (after) |
|---------------------:|----------------------:|----------------------:|
|                   90 |              2.15e-06 |           0 (exactly) |
|                  100 |              2.70e-02 |           0 (exactly) |
|                  104 |                   inf |           0 (exactly) |
|                  200 |                   inf |           0 (exactly) |

Measured against the hand-computed expectation, and not against the gap-0 answer, the gap-100 case
came out 2.7033e-02 off. That is 6.6% of the largest true log-probability (-0.40761), and it is the
"about 7% low at 100" that SushiBLAS's GPU_VALIDATION.md recorded for this defect before the fix.

The residual at the gaps that pass is absent from both columns, on both sides of the fix, because
it was never measured. A passing EXPECT_NEAR bounds it under 1e-4 and says nothing more. Do not
quote a figure for it. An earlier draft of the file header did, and the invented numbers survived a
full run unchallenged because nothing in the suite fails when a comment is wrong.

The file is a regression test and not an integration test because of how the defect fails. `+inf`
log-probabilities do not stop a run: the loss becomes `inf` or `nan`, the optimizer writes `nan`
parameters, and the training loop reports completion having learnt nothing. Nothing throws, and no
test that only checks "did it run" notices. This tier exists for that failure mode.

The expectations are hand-computed and not read off another path in this repository. Each row is
{a, a+1, a+2}, whose log-softmax is {-2 - k, -1 - k, -k} with k = ln(1 + e^-1 + e^-2), for any a.
The expected values are therefore the same for every gap under test, and any dependence on `a` is
the defect showing.

### ARowFarBelowTheBatchMaximumStaysFiniteAndCorrect

This test reproduces the defect itself. At a gap of 104 the old code returned `+inf` for row 1. At
100 it was 6.6% low on the largest log-probability (-0.43464 measured against a true -0.40761). At
90 it had drifted 2.15e-06 off its own gap-0 answer. The per-row maximum removes the gap from the
arithmetic, so the answer is the same at every gap, and the single shared expectation asserts that.

The tolerance in this test is 1e-4, which is wide enough to pass at gap 90 both before and after
the fix. That is intended. This case exists to catch `+inf` and gross error.
TheAnswerDoesNotDependOnTheGapAtAll makes the finer claim, that the gap changes nothing, at exact
equality.

### CrossEntropyOverSuchARowIsFiniteAndCorrect

This test checks the same defect one node downstream. Cross-entropy is `-sum(L * Y) / N` over the
log-probabilities, so an infinite row makes the loss infinite. The loss is the number the training
loop prints and the number every stopping rule reads.

The expectation is hand-computed: with the target on class 0 of both rows, the loss is
`-(2 * (-2 - k)) / 2 = 2 + k`, whatever the gap.

### TheCrossEntropyGradientOverSuchARowIsFinite

This is the failure that destroys a run, where the two earlier tests cover a bad reported number.
The fused cross-entropy gradient is `(exp(L) - Y) / N`, so an infinite `L` makes the gradient
infinite, the optimizer writes non-finite parameters, and every later step trains on NaN while the
loop reports success.

The expectation is hand-computed: `exp(L)` is the softmax of {a, a+1, a+2}, which is
{e^-2, e^-1, 1} / (1 + e^-1 + e^-2), the same for both rows. The test subtracts the one-hot target
and divides by the batch size of 2.

### TheAnswerDoesNotDependOnTheGapAtAll

This test states the property that makes the fix a fix and not a wider threshold: nothing in the
answer depends on the gap. Two batches differing only by a constant offset applied to one row must
produce identical log-probabilities for the row that did not move, and identical ones for the row
that did.

A shift by the global maximum fails this at any gap large enough to matter. A shift by each row's
own maximum passes it at every gap, because the gap never enters the arithmetic.

The test asserts equality and not a tolerance, and the arithmetic justifies it. Every logit in the
test is a small integer, so each row's own maximum is exact, the shift `x - row_max` is exact, and
both rows are reduced to the same three floats {-2, -1, 0} before anything transcendental happens.
The gap cannot reach the answer, so any difference is a real one. A tolerance here would have been
the same mistake in miniature: a threshold wide enough to hide the thing under test.

## test_loss_scaling.cpp

Suites: `Regression_LossScaling`.

Every failure of loss scaling is silent, which is what puts this part of mixed precision in the
regression tier.

- A scale that never backs off keeps overflowing, and every overflowed step is thrown away, so the
  loop stops learning while reporting an ordinary loss.
- A skip spelled as "multiply the step by zero" writes NaN into AdamW's first moment, because
  0 * inf is NaN and the gradient on a skipped step is an infinity by definition. Nothing throws;
  the loss goes NaN a step later, three layers from the cause. That is why the assertions in
  ASkippedStepLeavesTheParametersAndMomentsBitIdentical read the moments and not only the
  parameters. A test that checked the parameters alone would pass on this bug.
- An unscale that is off by a rounding is invisible against any tolerance, which is why
  UnscalingByAPowerOfTwoIsExact uses EXPECT_EQ.
- A device-resident scale that someone moved into Attributes would recompile the step on every
  backoff, converge identically, and cost the compile-once guarantee the architecture is built on.

The tests fall into five groups, in file order. AnFp16OverflowIsDetectedAndBacksTheScaleOff and
TheSameScaleInFloat32OverflowsNothing cover the overflow itself.
ASkippedStepLeavesTheParametersAndMomentsBitIdentical and
ACleanStepUnderTheSameScalerStillMovesEverything cover what a skipped step must leave behind.
UnscalingByAPowerOfTwoIsExact covers exactness. TheStepIsCompiledOnceThoughTheScaleChanges and
ABackingOffScaleStillCompilesOnce cover the compile-once guarantee with a scale that moves.
TheSeparateUpdateKernelRefusesAScaler covers what is refused.

### On constructing the overflow

This workload does not overflow on its own. Measured on this machine, fp16 without loss scaling
neither stalls nor diverges on it: both arms converge to the same loss. There is no natural
overflow to wait for, and a test that ran the demo with scaling on would prove nothing, since the
demo converged without scaling too. The tests therefore build the overflow. The run starts at a
scale of 2^20, which puts the fp32 logit gradient just past binary16's largest finite value of
65504. The cast into the backward matmul then produces infinities, which are real overflow and not
a simulation of it.

### K_OVERFLOWING_SCALE

The logit gradient is about 1/K_BATCH, so a scale of 2^20 puts it at roughly 65536 in fp32, just past
binary16's largest finite 65504. The backward matmul that sums over the batch takes it further. The
scale was chosen to be recoverable as well as overflowing. The run in
AnFp16OverflowIsDetectedAndBacksTheScaleOff has to reach a clean scale within its step budget,
because "the backoff never helped" and "the backoff was never given enough steps" are the same
observation and only one of them is a defect.

### train_scaled

The helper sets the scaler on the optimizer before it constructs the trainer, which records the
update into the plan. The screens have to be recorded ahead of the updates, and the optimizer owns
that ordering only once it knows about the scaler.

### AnFp16OverflowIsDetectedAndBacksTheScaleOff

This is the end-to-end case. A scale chosen to overflow binary16 produces infinities in the fp32
gradient buffers, count_nonfinite sees them, and the state machine backs the scale off until the
overflow stops.

The test then asserts that the run recovers. The backed-off scale has to reach the device, or every
step overflows for ever and the loop learns nothing while reporting a healthy loss. This assertion
pins the scale as a value the host rewrites between replays and not as a constant: a scaler whose
before_replay() did nothing would still back off on the host and would fail here.

The test asserts that the step counter does not advance through a thrown-away step. SGD's
first-step flag and AdamW's bias corrections are both derived from the counter.

The last assertion is that the final loss is finite. A loop that went NaN would report NaN there,
and no comparison against a threshold would catch a NaN.

### TheSameScaleInFloat32OverflowsNothing

This test is the control for AnFp16OverflowIsDetectedAndBacksTheScaleOff. Without it, "steps were
skipped" says nothing about fp16: an implementation that skipped every step would pass that test.
The same scale under the fp32 policy is nowhere near float32's range, so nothing may overflow and
nothing may be thrown away.

### ASkippedStepLeavesTheParametersAndMomentsBitIdentical

The gate exists for this assertion. A skip spelled through the existing scalars (step size zero,
beta1 one) looks like an identity and is not one: `m = 1*m + 0*inf` is NaN, and the optimizer never
recovers. So the test reads the moments as raw bytes before and after, and not only the parameters.

The test takes one healthy step first. The moments are then non-zero, and a NaN written into them
can be told apart from the zero they started at.

The poisoned gradient is non-finite, which is what an overflowed fp16 backward pass hands the
optimizer.

The test also asserts that the step counter did not move, so the next step's bias corrections are
the ones the skipped step would have used.

### ACleanStepUnderTheSameScalerStillMovesEverything

This test is the control for ASkippedStepLeavesTheParametersAndMomentsBitIdentical. With a finite
gradient the same machinery must move what the skipped step left alone, and the test asserts that
the parameter, the first moment and the step counter all changed. Otherwise an optimizer that had
stopped working altogether would satisfy "nothing changed".

### UnscalingByAPowerOfTwoIsExact

Scaling by a power of two changes only the exponent field, so the round trip is lossless and the
comparison is EXPECT_EQ. A tolerance would hide the defects this test exists to catch: a scale that
is not dyadic, or an unscale folded in after the moments when it belongs before them.

### TheStepIsCompiledOnceThoughTheScaleChanges

This test is the reason the scale is a device-resident value and not an Attributes slot. An
attribute is a compile-time constant baked into the recorded task, so every backoff and every
growth would recompile the step, and the loop would converge the same way while paying graph
analysis on each of them. The test's run is arranged so that the scale changes during it.

### ABackingOffScaleStillCompilesOnce

This test is the overflowing arm of the same guarantee. Backoffs are the direction that happens
under load, and they must not recompile either.

### TheSeparateUpdateKernelRefusesAScaler

The separate spelling has no branch to take, so it would unscale correctly and then apply every
overflowed step anyway. The optimizer refuses the combination by name and does not downgrade it
silently.

## test_mixed_precision.cpp

Suites: `Regression_MixedPrecision`.

Every guarantee in this file fails silently, which is what puts these tests in the regression tier
and not beside the integration tests. A run that reports success cannot catch any of them.

- A master weight retyped to fp16 does not throw. The optimizer keeps working, the loss keeps
  falling, and updates smaller than the weight's own fp16 spacing are discarded. That is the
  failure master weights exist to prevent, and it looks like a learning rate that wants tuning.
- A conversion placed on the wrong side of a reduction does not throw either; the model learns
  worse.
- The control matters as much as the case. If the fp32 policy also inserted casts, every "mixed
  inserted N casts" assertion in this repository would measure the pass's unconditional behaviour
  and not the policy's decision.

### optimizer_options

1e-3 is the reference AdamW default. It is not enough to move a model this small far in the couple
of hundred steps a test can afford, and a comparison between two runs that both barely moved would
agree for the wrong reason. 3e-3 is the demo's rate, so the numbers in this file and the ones in
docs/README.md come from the same setting.

### train

Everything except the policy is fixed, so a difference between two runs is the policy and nothing
else. The initialiser and the dataset are both seeded, and the seeds are the library defaults and
not values this file picked. A change to either default therefore shows up as a changed trajectory
in both arms, not as a silent divergence in one.

The helper reads the parameter dtypes after the run and not before. Fusion and the cast pass both
rewrite nodes, and a master weight that survived the passes could still have been swapped for a
narrow copy by the optimizer wiring.

### TheFp32PolicyIsIndistinguishableFromNoPolicyAtAll

This test is the control, in the strongest form available: zero casts, the same recorded operation
count, and the same losses to the bit. If the pass ever starts inserting a conversion
unconditionally, this test fails before any mixed-precision test can look correct for the wrong
reason.

### NarrowingCostsConversionsAndNothingElseStructural

Casts are nodes, so they show up as recorded operations, and the count has to be exactly the casts.
A higher count would mean a conversion landed inside a fusable chain and cost the fusion as well.
graph/cast_insertion.hpp's ordering argument exists to avoid that failure, and no other test in
this file would notice it.

### MixedAndSinglePrecisionTrainToTheSamePlace

This test makes the numeric claim, and its tolerance is derived, not tuned until green. fp16
carries 11 bits of significand, so one rounded value is accurate to about 2^-11 = 4.9e-4 relative.
A step's loss passes through a handful of rounded values, so a few times that figure is the
expected deviation.

The measured figure on `sa demo mlp --epochs 8 --batches 16` (a wider model than this one, 128
steps) was 9.5e-4 relative, at the end of the first epoch. That is the largest loss whose six
printed decimals still carry enough digits to divide. The test's bound is 2e-2, about twenty times
the measured figure, and it is a bound and not a measurement. It was chosen so the test survives a
change of seed or schedule. Passing it says the deviation is under 2e-2 and nothing more precise.

The test first asserts that both runs learned. Two runs that both diverged would agree and the
agreement would mean nothing.

### AGeluLayerIsNotFoldedUnderANarrowingPolicy

graph/precision.hpp's rules 2 and 5 meet in this test, and it asserts that they meet consistently.

Rule 2 says a fused epilogue runs inside the GEMM's kernel in the GEMM's element type, so an
activation the catalog folds has to compute in the GEMM's type or fusion would change the numbers.
Rule 5 says GELU stays wide: the tanh approximation computes `x*x*x`, which overflows fp16 at |x|
about 40.3. Since FusedForm::GemmBiasGelu exists, the two rules would contradict each other if
nothing kept them apart.

The pass order keeps them apart, and it does so by structure. The cast pass runs before fusion, so
an activation the policy keeps wide gets a K_CAST between it and the narrowed K_BIAS_ADD, and a chain
with a conversion in the middle matches no catalog entry. Under fp32 the layer folds whole. Under a
narrowing policy it falls back to gemm+bias and the GELU runs on its own, as it did before the
entry existed.

If this test ever sees a gemm+bias+gelu node, an fp16 `x*x*x` is being computed, and nothing else
in the suite would say so.

The test also asserts that fusion did not stop altogether. The bias still folds on both layers, so
the fallback is the shorter catalog entry and not an unfused graph.

The last assertion is that the K_GELU is still in the graph and still in the master type. That is the
other half of rule 5 and the reason the fold is refused.

## test_static_memory.cpp

Suites: `Regression_StaticMemory`.

The file tests four guarantees of the static memory planner whose failure would be silent: the loop
still converges, the run still reports success, and nobody notices.

1. A training loop allocates nothing after the first step. The test counts this. Every Storage
   allocates through the runtime context, so `RuntimeContext::live_usm_count()` is the
   live-allocation count, and one extra buffer moves it by exactly one.
2. Packing shrinks the footprint. A pass that quietly stopped packing would leave every other test
   in the suite green. The footprint is measured on the unfused graph, because fusion reaches the
   same bytes by a different route: it deletes the intermediates where packing overlaps them. A
   planner test that let fusion do its work first would be measuring the catalog.
3. Packing changes no number. If it does, two values are aliasing that should not be, and the right
   response is to find them, not to record the reference again.
4. A buffer the planner packed is rewritten every replay. A K_ZERO buffer that is filled once and
   then only read holds zeros for exactly one step and then whatever overwrote it. Such a model
   trains, badly, for ever.

### train

With `packed` false, the pool has already adopted the empty plan, which covers no value and asks
for no arena. That is the behaviour the library had before the planner pass existed, which makes it
the only honest control.

The `fusion` parameter says what the fusion pass is allowed to do. Fusion is a confound in these
measurements: a fused subtree's intermediates stop being values, so fusion removes exactly the
buffers packing used to overlap. Every test in this file therefore says which graph its claim is
about.

### ATrainingLoopAllocatesNothingAfterTheFirstStep

This is the headline property of the planner pass, and the test counts it. The baseline is taken
after the first step, so a one-time lazy cache inside the runtime is tolerated while a per-step
allocation is not. A per-step buffer would still train correctly and would show up only as a slowly
growing process.

### PackingShrinksTheFootprintOfTheSameModel

The footprint is measured on the unfused graph on purpose. Fusion is a different mechanism reaching
the same bytes: a fused subtree's intermediates stop being values, so they need no slot where
packing would give them a shared one. Planning what fusion already deleted would make this test a
statement about the fusion catalog and not about the planner, and it would stop testing anything
the day a new catalog entry landed.

### FusingTakesTheBuffersPackingUsedToOverlap

This test states the interaction between fusion and packing so that nobody has to rediscover it. On
the fused graph the planner's own report reads like a broken planner: every remaining activation's
live range intersects every other's, so the planner overlaps nothing and prints "0% saved". The
planner is not broken. The buffers it used to overlap no longer exist, because a fused subtree's
intermediates are not values any more. That is the better outcome of the two, and it is why the
arena has nothing left to do.

`unpacked_bytes` measures this directly. It is what the same tensors would cost at one allocation
each, so a drop in it means buffers disappeared and were not merely shared.

Against the packed-but-unfused graph the fused footprint is never worse. On this model it is
exactly equal, because what fusion deletes here happens to be what the planner was overlapping. The
assertion is written as `<=` and not `==` on purpose: the equality is a property of this model's
shape and not of the two passes, and pinning it would fail on the next catalog entry for a reason
nobody should have to investigate.

### PackedAndUnpackedTrainingAgreeByteForByte

If the two runs ever differ, two tensors are sharing bytes they should not. The fix is to find
which two, never to relax the comparison. The test compares bytes and not a tolerance for the
reason test_determinism.cpp gives: a tolerance cannot tell a rounding difference from a real
overlap.

### APackedZeroBufferIsRefilledOnEveryReplay

K_ZERO is the one operation whose output nothing else writes. It would work perfectly if it were
filled once outside the step, until the planner handed its bytes to a later value. The test
corrupts the buffer from the host between two replays, which is what a reused arena slot does, and
the second replay must still see zeros.
