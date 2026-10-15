# tests/layers/train/test_trainer/test_fused_optimizer.cpp


Suites: `Medium_FusedOptimizer`, `Medium_FusedOptimizerDeathTest`.

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

## Why byte for byte is the right bar

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

## The second half: folding the update into the compiled plan

Fusing makes the update replay-invariant, and replay-invariance is what lets the update be recorded
into the same graph as the forward and backward passes. That is a separate change with its own

silent failure mode. A dependency edge the runtime does not see would let the update race the
backward pass that produces its gradients, and a race that usually resolves the right way looks
like training. So the folded and unfolded placements are compared byte for byte too, and
compile_count() is pinned at one with the optimizer inside the plan.

## K_ELEMENTS

`K_ELEMENTS` is 133, which is not a multiple of the 128-bit vector width. Both the wide pass and the
scalar tail of Launch::zip_inplace therefore run on every call.

## gradient_for

The first four entries of the gradient are placed by hand and not sampled: an exact zero, a
negative zero, a denormal-adjacent tiny value and a large one. Zero is where AdamW's
`0 / (sqrt(0) + eps)` lives. Negative zero distinguishes `b = staged` from `b = 0*b + 1*staged`:
the second form turns -0.0 into +0.0, and no tolerance would see that. The two extremes exercise
the square root and the division away from unity.

## drive

The helper uses no graph, no trainer and no device RNG. It has one parameter, one gradient
rewritten from the host between steps, and the optimizer's own step(). The update is then the only
thing in the measurement.

## train

The helper builds a fresh Engine, because reproducibility is a property of one engine. dataset.hpp
explains the RNG offset counter behind that.

## AdamWWithoutDecayTakesTheOtherBranchAndStillMatches

With `weight_decay = 0` the decay factor is exactly 1, and both paths must skip the multiplication
and not multiply by 1. `theta * 1.0f` happens to be bit-identical here, but the skip is a separate
branch in each path and this test is what covers it.

## SGDsFusedKernelMatchesTheSeparateOperationsExactly

The test runs every branch of the SGD recipe, because each one is a separate line of the fused
functor and a separate chance to have written it differently.

## TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt

If the two runs diverge, the update is not ordered against the graph the way the separate flush
ordered it. The runtime is then missing a dependency edge, and the fix is that edge, never a
tolerance.

## TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo

This is the end-to-end version of AdamWsFusedKernelMatchesTheSeparateOperationsExactly. That test
drives one optimizer over a synthetic gradient. This one runs the whole pipeline, with gradients
the backward pass produced and a trajectory long enough for a one-ULP difference in an early
denominator to become a visible one.

## TheStepCompilesOnceWithTheOptimizerInsideIt

The model has four parameters, so four fused updates joined the plan. The test states the count as
a difference and not as an absolute, so a change to the graph itself does not have to be recorded
again here.

## TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve

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

## AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored

The separate spelling bakes the current step's bias corrections into its recorded tasks. Folded
into the plan, it would replay step one's corrections for ever and still converge. That is why the
request has to be an error and not a quiet fallback.

## SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused

Before the update joined the plan, `trainer.run_step()` called `optimizer.step()`, and a caller who
also called it got a visible double update. With the update inside the replay and invisible from
the outside, the same mistake would train at twice the learning rate and produce a run that looks
fine and is wrong.

## TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored

The momentum buffer is allocated at construction or not at all, so the fused kernel's operand count
is fixed there too. Before this check existed, the separate path dereferenced a null tensor and the
fused path would have trained without momentum.
