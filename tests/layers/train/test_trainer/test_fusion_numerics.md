# tests/layers/train/test_trainer/test_fusion_numerics.cpp


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
AReLUMLPTrainsIdenticallyFusedAndUnfused, ATanhMLPTrainsIdenticallyFusedAndUnfused and
ASigmoidMLPTrainsIdenticallyFusedAndUnfused. The second is the five GELU tests, from
AGELUMLPTrainsIdenticallyFusedAndUnfused to DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt,
which the section on the spilled activation describes. The third counts what fusion cost:
TheFusedStepIssuesTenFewerKernelsPerStep, AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias,
WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt and
DisablingOneCatalogEntryLeavesTheRestAlone. The fourth covers the dynamic selector, from
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove to EveryChoiceCarriesAReasonSomebodyCanRead.

## Why byte for byte is the right bar

The argument is the one test_fused_optimizer.cpp makes for the optimizer. Every element is computed
by the same operations in the same order. The one plausible source of a moved bit is a compiler
contracting `a * b + c` into an FMA inside one kernel where it could not across two, and
`-ffp-contract=off` rules that out; the whole stack inherits the flag through SushiRuntime's
`sushiruntime_deterministic_fp` contract. Each fused functor is written in the operand order of the
calls it replaces, and a constant that reached a kernel through a `float` parameter still does.

If a comparison fails, the response is to find which expression moved, not to widen the comparison.

## The second half: the activation that needs a spill to be fused

GELU's gradient differentiates at its input, so the epilogue that folds GELU has to write that
input to an auxiliary buffer on its way past. cuBLASLt calls this GELU_AUX_BIAS; here it is
FusedForm::GEMMBiasGELU. Every way of getting it wrong is silent, and worse than the other
activations: a forward pass that matches perfectly while the gradients are taken at the wrong point
still converges, to a different place.

A forward comparison is therefore not the test. The test is that a GELU model trained end to end
reaches byte-identical parameters after 24 optimizer steps. That is a statement about the
gradients, since nothing else moves a weight. It is taken three ways, against three different
fillings of the same buffer:

- AGELUMLPTrainsIdenticallyFusedAndUnfused compares against the graph with no fusion pass at all.
- DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt compares against the same graph with only
  this catalog row switched off, so the difference is the entry and nothing else.
- ASelectorRefusalStillFillsTheSpillAndMovesNoBit compares against a selector that declines the
  fused arm, so the aux buffer is filled by a separate copy in the lowering and not from inside
  the epilogue.

TheSpillHoldsThePreActivationItself then reads the buffer directly and compares it with a
host-computed pre-activation, because three spellings can agree while all three hold the wrong
tensor.

## train

The helper builds a fresh Engine, because reproducibility is a property of one engine; dataset.hpp
explains the RNG offset counter behind that. It trains with AdamW and not plain SGD because AdamW's
division by an accumulated square root is where a one-ULP difference stops being one ULP.

## pretend_device

The one device on the test machine is an OpenCL CPU with no dedicated scratchpad, and it supports
doubles. A selector rule that can only be exercised on hardware nobody owns is a rule nobody has
run, so the declining branches are driven from a described device.

## AReLUMLPTrainsIdenticallyFusedAndUnfused

This is the demo's own model. Both linear layers fuse, the first with its ReLU and the second
without one, and so does the cross-entropy gradient's elementwise tail. One run therefore covers
three separate collapses that reach the same arithmetic.

## ATanhMLPTrainsIdenticallyFusedAndUnfused

tanh's gradient reads its own output, so the pre-activation is dead and the whole layer folds. The
tanh epilogue is a different template instantiation from ReLU's, which is why it has its own
comparison.

## AGELUMLPTrainsIdenticallyFusedAndUnfused

This test runs the GELU hazard end to end, and its claim is about the gradients and not the forward
pass. A K_GEMM_BIAS_GELU that computed the right activation and spilled nothing would match the
unfused forward exactly and still differentiate at the wrong tensor. Nothing would throw, no value
would go non-finite, and the loss would still fall. What such a kernel could not do is land the
parameters on the same bits after 24 AdamW steps, because a weight only moves by its gradient.

The fused graph holds two fused linear layers, the first folding its GELU and the second with no
activation to fold, and the fused cross-entropy gradient tail.

The test also asserts that the loss fell. Without that assertion, "identical" could describe two
runs broken in the same way.

## TheGELUFoldIsReportedAsAFoldAndNotAsARefusal

The test asserts that the spilled value is the one the backward rule reads. It checks this on the
IR and not through a loss, because "the gradient is taken at the right tensor" has no numerical
symptom when it is wrong.

## TheSpillHoldsThePreActivationItself

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

## ASelectorRefusalStillFillsTheSpillAndMovesNoBit

This test is the control for TheSpillHoldsThePreActivationItself. With the fused arm declined, the
aux buffer is filled by a separate copy in `record_fused_gemm` (`modules/core/graph/source/lowering/fused.cpp`) and
not by the epilogue.
That is a different code path to the same buffer, and the gradients have to come out identical.
Deleting either write makes this comparison fail.

## DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt

This is the A/B comparison the catalog entry was written against: the same model with the catalog
row on and off. With the row off, the K_GELU is an ordinary IR node and the pre-activation is an
ordinary value. With it on, the pre-activation is a second output of a fused GEMM, written from
inside a device kernel. The parameters after 24 steps have to be the same bits, and they are the
same bits only if the gradients were.

Disabling the form costs two more kernels, as switching the ReLU form off does. The activation
becomes an out-of-place IR node again, which is a copy and a gelu. The spill costs nothing on the
unfused side, because the value the fused node spilled is the value K_BIAS_ADD writes anyway.

## TheFusedStepIssuesTenFewerKernelsPerStep

This test counts the headline claim and does not time it. A wall clock says different things on a
loaded machine, and the kernel count is the number the change controls.

The saving is ten for this model. The first layer's gemm+bias+relu replaces five calls (gemm; copy
and ger for the bias; copy and relu) with one. The second layer's gemm+bias replaces three with
one. The cross-entropy gradient's exp/sub/scale replaces five with one.

The saving has no cost beside it. Each fused GEMM needs no host-side ordering node to carry
the dependency edges execute_gemm could not declare for a bias its functor dereferences. Two layers
meant two extra nodes. SushiBLAS's seam now takes the epilogue's own access set, so
`operations_per_step` is the whole plan again and the two counts cannot drift apart.

## AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias

The fused spelling carries a hazard the unfused one does not. The epilogue dereferences the bias
from inside the GEMM kernel, and the optimizer rewrites that same bias inside the same compiled
plan. SushiBLAS cannot see into the functor, so this library has to make the declaration:
`gemm_epilogue(f).reads(bias)` in `record_fused_gemm`.

The edge is what the test is for, and a loss cannot stand in for it. With no edge the two tasks may still come out
in the right order on any given run, so a falling loss proves nothing about ordering, and this
failure mode survives an otherwise green suite. The case reads the edge from
`SushiRuntime::Graph::plan()`, as the predecessor count of the second recorded node. The node
count is asserted for a related reason:
the plan holds two nodes, and no host-side ordering node carries this edge.

The bias-writing task in the test stands in for the optimizer's update: a task in the same plan
that writes the bias the epilogue reads. It is recorded first, so an edge can only run from it to
the GEMM.

## WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt

This test covers the other half of the epilogue's access set, the half only the spilling arm has.
`.reads(bias)` orders the GEMM behind whoever writes the bias; `.writes(aux)` orders whoever reads
the spill behind the GEMM. Without the second declaration, the backward pass that differentiates at
the pre-activation races the kernel that produced it. A race that usually resolves the right way
cannot be told apart from training, so a loss cannot stand in for the edge here either. The case
asserts the predecessor count of the reader in the plan.

The test emits a reader of the pre-activation after the K_GELU, the way the differentiation pass
emits gelu_backward.

## DisablingOneCatalogEntryLeavesTheRestAlone

The per-entry switch serves two callers. TheFusedStepIssuesTenFewerKernelsPerStep needs it as its
A/B control, and a user who has measured a different result on their own model needs it as a way
out. Switching the three-op GEMM form off must fall back to the two-op form and not to nothing.

The catalog refusal costs two kernels more and not one, and the two ways of declining differ here.
Refused at the catalog, the ReLU stays an ordinary IR node. An IR activation is out of place, so it
lowers to a copy and a ReLU. Refused by the selector, as in
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove, the node is already fused and its output
tensor is the activation's own result, so the trailing ReLU runs in place and costs one. The
numbers are the same either way; the launch counts differ.

## ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove

The seam between the two stages makes one promise. The static pass decides what could be fused and
the selector decides what is fused, so a device that wants no fusion gets an unfused plan out of a
fused graph. The numbers stay the same because every spelling of a fused node is the same
arithmetic.

## AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory

Under ForceReference the epilogue is invoked on the host by Kernels::Reference::gemm,
so any pointer it carries must be host-readable there. SushiAI's own allocations are shared USM and
would be readable. A bound allocation need not be, and nothing in the selector can ask which kind
it got.

The zip-based form runs no host arm, so it is unaffected. A blanket refusal of every form in these
modes would be a different bug from the one being prevented.

## EveryChoiceCarriesAReasonSomebodyCanRead

A selector nobody can interrogate is a selector nobody will trust. The static pass carries a report
for the same reason: both of its answers produce a plan that trains.
