# tests/core/graph/test_fusion.cpp


Suites: `Small_Fusion`.

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

The activation whose gradient differentiates at its input is not on that list.
K_GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal is a condition: an INPUT-gradient activation may be folded only by an
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

## EveryFormRoundTripsThroughItsOperationIdentity

The lowering dispatches on `fused_form_op` and `fused_form_of`. A form that does not round-trip
would lower as "no recorder for this operation", which is loud, but only at run time and only on
a graph that happens to contain it.

## AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts

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

## AMatmulAndABiasAddBecomeOneNode

`report.fused().size()` is 1, and the fused node reads A, B and the bias and still writes the
value the caller named. That is the whole reason ids are preserved.

The `flag(0)` and `flag(1)` checks hold the transposition flags. They have to survive, because
the fused recorder hands them straight to `execute_gemm`, and losing one gives a wrong answer of
exactly the right shape.

## TheLongestMatchWinsOverTheShorterOneInsideIt

Both gemm+bias and gemm+bias+relu match this graph. Rooted at the bias_add, the shorter one
would win and the relu would be left outside for good, which is why the matcher walks roots in
reverse.

## TheCrossEntropyGradientsElementwiseTailBecomesOnePass

With `scale(sub(exp(log_probs), targets), 0.25)` observed, `fuse` reports one
`FusedForm::ExpSubScale`. The node that produces the scaled value reads `log_probs` and
`targets` and has the attribute 0.25, and the two intermediate values have no producer.

## ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

## AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the K_MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

## AnIntermediateTheCallerReadsByIDIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(K_GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
bias_add, whose own intermediate is free.

## AParameterIsProtectedWithoutBeingListed

A persistent value's buffer outlives the step whatever the graph says, so the caller must not
have to remember to list it. In this graph the "intermediate" is a parameter. That is an odd
graph, but it is the shape a caller would get wrong.

The assignment `lifetime = OPTIMIZER_STATE` forces the exp's result into a persistent class, as
a checkpointed activation would be.

## AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills

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

## ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the K_GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the K_GELU's position, so a reader
of the pre-activation sitting between the bias_add and the K_GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(K_GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

## ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The K_GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

## TheGELUFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The K_MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

## DisablingTheGELUFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::Disabled`, is the only one.

## ARecoverableActivationNextToAFusedGEMMIsFoldedNotReported

This case mirrors DisablingTheGELUFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

## DisablingThePassLeavesTheGraphExactlyAsItWas

With `enabled` false over a linear trace followed by a ReLU, the report is not enabled and
lists nothing fused. The graph keeps its node count, its value count and its one MATMUL.

## DisablingOneFormFallsBackToTheShorterOne

With `FusedForm::GEMMBiasReLU` disabled, the report counts no `GEMMBiasReLU` and one
`GEMMBias`, and holds a decline of `FusionDecline::Disabled`.

## ValueIDsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

## TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

## TheReportNamesEverySubtreeItCollapsed

After a default `fuse`, `to_string` holds "gemm+bias+relu" and "dropping value", and
`nodes_after` equals the graph's node count.

## StateLeavesAndTheRefusalMarkSurviveTheRewrite

The graph is a K_MATMUL, a K_BIAS_ADD whose bias is a buffer, and a K_RELU. The weight and the buffer
are bound to the states 21 and 22 and the graph is marked refused before `Graph::fuse` runs.
The report lists one fused form and the graph holds one K_GEMM_BIAS_RELU node, so the pass did
rebuild the graph. Both `state_leaf` lookups and the whole `state_leaves()` table are what they
were. The buffer keeps the lifetime `BUFFER` and no gradient, the weight still requires one,
and `has_refusal()` is still true.
