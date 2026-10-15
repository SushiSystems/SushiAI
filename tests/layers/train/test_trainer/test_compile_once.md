# tests/layers/train/test_trainer/test_compile_once.cpp


Suites: `Regression_CompileOnce`.

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

Each test in this file spells out its own setup and shares none of it through a fixture struct.
Trainer holds raw pointers into the pool and the optimizer, so the construction order (create the
state, bind, then initialise, then optimizer, then trainer) is part of the contract, and a test that spells the order
out fails loudly if that order ever stops working. The setup matches
tests/layers/train/test_trainer/test_convergence.cpp line for line for the same reason.

## ManyFitCallsCompileTheStepExactlyOnce

`sa demo mlp` drives one fit() per epoch so it can print a per-epoch trajectory. That call pattern
differs from a single fit() with epochs=N. The integration test calls fit() once, so it could not
see a recompile hidden inside fit().

## EveryReplayDispatchesRatherThanReportingCompleteImmediately

This test looks at a replay that dispatches nothing from the outside. Such a run() returns
successfully, so the parameters never move and the loss goes flat. It shows from the second
replay onward when the first replay still runs off the state the compilation left.

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

The replay loop repeats the check three times. Each pass poisons the output again, changes the
input, and requires the output to track the new input and not the old answer or the poison.

## CompileCountSurvivesEverythingTheLoopDoesBetweenSteps

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
