# tests/layers/train/test_trainer/test_compile_once.cpp


Suites: `Medium_CompileOnce`.

This file was the regression tier's first. It pins the two claims in this library whose failure is
completely silent.

"Trace once, differentiate once, compile once, replay for ever" is the premise the design is
arranged around. Both halves of it fail without a symptom:

- A trainer that recompiles per step still converges. It pays graph analysis every step, and
  nothing in the output says so.
- A replay that never dispatched still reports complete. A plan whose per-run state is not
  restored before a replay has its completion sentinel ready at once, so run() returns having
  done nothing. The loss stops moving, which reads like a bad learning rate.

SushiRuntime restores that state inside `Graph::run()`, and sushiblas cannot see a training step
at all. So both claims are pinned here, in the shape the trainer and `sa demo mlp` use.

The fixture `Medium_CompileOnce` holds the classifier, its state, its pool and its parameter slots,
and `prepare()` traces, binds and initialises them in that order. Trainer holds raw pointers into
the pool and the optimizer, so each test makes its own optimizer and trainer after `prepare()`,
and the order (state, bind, initialise, optimizer, trainer) stays in the test where a reader sees
it. The helpers `fit_once`, `step_moves_the_parameter`, `round_of` and `replay_over` take the
repeated step of a test, so no test holds a loop or a branch.

## ManyFitCallsCompileTheStepExactlyOnce

`sa demo mlp` drives one fit() per epoch so it can print a per-epoch trajectory. That call pattern
differs from a single fit() with epochs=N. The integration test calls fit() once, so it could not
see a recompile hidden inside fit(). Six explicit calls of `fit_once` each expect a compile count
of one in the report and in the trainer, and the steps they ran add up to 6 times the batches. This
file is the one home of the promise; the copy in `test_convergence.cpp` is deleted.

## EveryReplayDispatchesRatherThanReportingCompleteImmediately

This test looks at a replay that dispatches nothing from the outside. Such a run() returns
successfully, so the parameters never move and the loss goes flat. It shows from the second
replay onward when the first replay still runs off the state the compilation left. One check per
replay, five in all: the step over that batch changes the first parameter. The old case compared
each loss with the one before it, which a changed loss does not prove.

## TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline

This test catches a run that dispatches nothing at the first run, where
EveryReplayDispatchesRatherThanReportingCompleteImmediately catches it only at the second replay.

That other test reads the defect off a loss that stops changing, so it needs two replays to see
it. A first run over a plan with an empty baseline fails the same way: the completion sentinel is
ready before anything is dispatched, and the run returns success having run no kernel. The only
thing that distinguishes that from a step that ran is the contents of the output buffer.

So the test first poisons the buffer with a value the graph cannot produce, and it asserts that the
poison is gone. A run that dispatched nothing leaves the poison where it was. The step reports
`compile_count()` 1 before that first run and still 1 after the four runs.

The graph doubles its input, which is exact in float, so the four results are compared exactly:
{2, ..., 12} for the inputs 1 to 6, then 22 to 32, 42 to 52 and 62 to 72 for the inputs that
follow in steps of 10. Each call of `replay_over` poisons the output again, changes the input, and
requires the output to track the new input and not the old answer or the poison. The old case
compared with a tolerance of 256 unit roundoffs; an exact doubling needs none.

## CompileCountSurvivesEverythingTheLoopDoesBetweenSteps

ManyFitCallsCompileTheStepExactlyOnce asserts compile-once for the plain path. This test pins that
none of the things a real loop does between steps reaches back into the plan: rewriting the batch
buffers, interleaving an optimizer that records and flushes its own graph every step, stopping and
resuming, and changing a hyperparameter mid-run.

The test builds two datasets of the same shape with different contents. The compiled plan reads its
inputs by address, so swapping the data a step trains on must be a copy into those buffers and
nothing more.

Four calls of `round_of` stop and resume: fit() returns and is entered again, which is the pattern
`sa demo mlp` uses to print a per-epoch trajectory. The fits alternate the two datasets, and each
round runs one single step over each dataset's batches 0 and 1.

Each round also halves the learning rate. A learning-rate schedule changes a host-side
scalar the optimizer reads per step. If that scalar ever became a compiled attribute it would force
a recompile; ir.hpp's warning on Attributes says the same.
