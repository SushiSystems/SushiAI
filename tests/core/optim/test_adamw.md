# tests/core/optim/test_adamw.cpp


Suites: `Small_AdamW`, `Small_AdamWKernel`, `Small_AdamWOptions`.

Tests what `check_adamw_options()` refuses, the state an `AdamW` starts from and its first update.

The optimizer is built on a `HostExecutionBackend`, which allocates the moments and replays the
update on the host, so the file is small. `test_optimizer.cpp` covers weight decay per slot,
`test_run_state.cpp` the names of the moments in a run state, and
`tests/layers/train/test_trainer/test_fused_optimizer.cpp` the fused kernel against the separate
operations.

## Small_AdamW

The fixture's `make_slots` returns one slot "weight" of three elements at 1.0, with the gradient
2.0, -3.0 and 0.5.

## TheReplayFollowsTheKernel

Rows over `UpdateKernel`: an AdamW reports itself replay invariant with the fused kernel and not
with `UpdateKernel::Separate`.

## TheCheckRefusesOptionsOutsideTheirRange

Rows over the options: a `beta1` of 1.0 and a `beta2` of -0.1 each fail with
`InvalidDescription` naming the option and "[0, 1)". An `eps` of 0 fails with "eps must be
positive".

## MakeRunsTheOptionsCheck

`make()` with a `beta1` of 1.0 fails with the same refusal as the check.

## TheMomentsStartAtZero

Both moments of the one parameter hold zeros after `make()`.

## TheFirstStepMovesEachElementByTheRate

With a rate of 0.1 and no decay, one step moves each element by 0.1 against the sign of its
gradient, within 1e-6. On the first step the bias-corrected moments are the gradient and its
square, so the update is the gradient over its own size plus `eps`.

## TheFirstStepFillsTheFirstMoment

After one step the first moment is 0.1 times the gradient, within 1e-6.

## AMomentPastTheLastParameterIsFatal

`first_moment(1)` on an optimizer of one parameter stops the process with a report naming
"updates 1 parameters" and "no moment 1".
