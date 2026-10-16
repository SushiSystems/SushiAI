# tests/core/optim/test_adamw.cpp


Suites: `Medium_AdamW`.

Tests what `AdamW::make()` refuses, the state it starts from and its first update.

The optimizer allocates its moments through an engine and runs its update on the device, so the
file is medium. `test_optimizer.cpp` covers weight decay per slot, `test_run_state.cpp` the
names of the moments in a run state, and
`tests/layers/train/test_trainer/test_fused_optimizer.cpp` the fused kernel against the separate
operations.

## Medium_AdamW

The fixture's `make_slots` returns one slot "weight" of three elements at 1.0, with the gradient
2.0, -3.0 and 0.5.

## TheTypeAndReplayFollowTheKernel

An AdamW names itself `K_ADAMW_TYPE`. It reports itself replay invariant with the fused kernel
and not with `UpdateKernel::Separate`.

## MakeRefusesOptionsOutsideTheirRange

A `beta1` of 1.0 and a `beta2` of -0.1 each fail with `InvalidDescription` naming the option
and "[0, 1)". An `eps` of 0 fails with "eps must be positive".

## TheMomentsStartAtZero

Both moments of the one parameter hold zeros after `make()`.

## TheFirstStepMovesEachElementByTheRate

With a rate of 0.1 and no decay, one step moves each element by 0.1 against the sign of its
gradient. On the first step the bias-corrected moments are the gradient and its square, so the
update is the gradient over its own size plus `eps`. The first moment is 0.1 times the
gradient. Both are compared within 1e-6.

## AMomentPastTheLastParameterIsFatal

`first_moment(1)` on an optimizer of one parameter stops the process with a report naming
"updates 1 parameters" and "no moment 1".
