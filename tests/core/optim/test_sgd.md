# tests/core/optim/test_sgd.cpp


Suites: `Medium_SGD`.

Tests what `SGD::make()` refuses and the update it applies with and without momentum.

The optimizer runs its update on the device through an engine, so the file is medium.
`test_optimizer.cpp` covers weight decay per slot, `test_run_state.cpp` the momentum buffers in
a run state, and `test_gradient_clip.cpp` the clipped update.

## Medium_SGD

The fixture's `make_slots` returns one slot "weight" at 1.0 per element with the gradient a case
passes. `step_once` steps the optimizer, waits for the device and returns the parameter.

## TheTypeAndReplayFollowTheKernel

An SGD names itself `K_SGD_TYPE`. It reports itself replay invariant with the fused kernel and
not with `UpdateKernel::Separate`.

## NesterovNeedsMomentumAndNoDampening

Nesterov with momentum 0 fails with `InvalidDescription` and "positive momentum". Nesterov with
momentum 0.9 and dampening 0.1 fails with "undefined with dampening".

## APlainStepSubtractsTheScaledGradient

With a rate of 0.5 and no momentum, one step over the gradient 2.0, -1.0 and 0.5 leaves each
element at 1.0 minus 0.5 times its gradient, within 1e-6.

## MomentumAccumulatesFromTheSecondStep

With a rate of 0.1, momentum 0.9 and a gradient of 1.0, the buffer is the gradient after the
first step and 0.9 times the buffer plus the gradient after the second. The parameter is
therefore 0.9 and then 0.71, each within 1e-6.
