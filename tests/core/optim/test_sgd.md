# tests/core/optim/test_sgd.cpp


Suites: `Small_SGD`, `Small_SGDKernel`.

Tests what `SGD::make()` refuses and the update it applies with and without momentum.

The optimizer is built on a `HostExecutionBackend`, which allocates its state and replays the
update on the host, so the file is small. `test_optimizer.cpp` covers weight decay per slot,
`test_run_state.cpp` the momentum buffers in a run state, and `test_gradient_clip.cpp` the clipped
update.

## Small_SGD

The fixture's `make_slots` returns one slot "weight" at 1.0 per element with the gradient a case
passes. `step_once` steps the optimizer and returns the parameter.

## TheReplayFollowsTheKernel

Rows over `UpdateKernel`: an SGD reports itself replay invariant with the fused kernel and not
with `UpdateKernel::Separate`.

## NesterovNeedsMomentum

Nesterov with momentum 0 fails with `InvalidDescription` and "positive momentum".

## NesterovNeedsNoDampening

Nesterov with momentum 0.9 and dampening 0.1 fails with `InvalidDescription` and "undefined with
dampening".

## APlainStepSubtractsTheScaledGradient

With a rate of 0.5 and no momentum, one step over the gradient 2.0, -1.0 and 0.5 leaves each
element at 1.0 minus 0.5 times its gradient, within 1e-6.

## MomentumStartsFromTheGradient

With a rate of 0.1, momentum 0.9 and a gradient of 1.0, the buffer is the gradient after the first
step, so the parameter is 0.9, within 1e-6.

## MomentumAccumulatesFromTheSecondStep

The buffer is 0.9 times the first buffer plus the gradient after the second step, so the parameter
is 0.71, within 1e-6.
