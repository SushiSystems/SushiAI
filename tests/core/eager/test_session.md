# tests/core/eager/test_session.cpp


Suites: `Medium_EagerSession`.

Tests the steps a `Session` runs and the points at which a `Step` runs them.

This file needs an engine. Its cases are the acceptance list of
`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`, section 10, as far as one module reaches.

## eager_run_support.hpp

`Support::Loop` owns a `ScopePartitioner`, a `RegionExecutor`, a `Session`, a model of linear
layers and one batch. `step(options)` traces the model, differentiates, applies the update and
reads the loss. `TraceOptions::scoping` says where scopes open: nowhere, one per layer, one
between the first layer and its activation, or "encoder" and "head". `PlainDescent` is the
update: one `axpy` of minus the rate times the gradient per parameter. It is a
`Graph::StepRecorder` and counts its recordings and its replays. `make_model` fills every tensor
with `fill_wave`, so two models of the same widths start from the same numbers.

## Compiled

`make_compiled` traces the same two-layer model in the same order, differentiates it, runs the
cast and fusion passes with the leaves, the loss and the gradients observed, plans the memory
with the loss and the gradients pinned, binds the same tensors and compiles the graph with a
`PlainDescent` behind it through `Graph::CompiledStep`.

## AnUnscopedStepGivesTheLossBytesOfTheCompiledStep

Four steps of both paths give the same loss bytes. With no scope the eager step is one region
that holds the forward and the backward nodes, so the two paths lower the same nodes. The case
compares with `Graph::CompiledStep` and not with `Train::Trainer`, which a test of this module
cannot name.

## ASettledLoopComposesOnceAndAllocatesNothingMore

A three-layer model with one scope per layer runs five steps. After the first, the composition
count stays 1 and the lowering count, the pool bytes, the session's allocation count and its
tensor count do not move. The update is recorded once and replayed five times.

## FlippingABranchRebuildsThatScopesTwoRegionsAndNoOther

Two steps with a K_TANH head, then one with a K_RELU head. The head's forward and backward
regions are lowered twice and every other region once. The loss is traced after the head closes,
in the second unnamed segment of the step, so its region is named "#1"; the first unnamed
segment holds the input leaf and no region.

## AScopeBetweenALinearLayerAndItsActivationTrainsToTheSameLoss

Two loops start from the same numbers, one with no scope and one with a scope between the first
layer and its activation. Their losses are compared within `K_LOSS_TOLERANCE`, 1e-5, over five
steps, and not byte for byte: the two loops run different kernels for the first layer.
No region of the scoped loop reports a `GEMMBiasReLU`.

## AScopeThatAppearsAndDisappearsRecordsTheRegionsItRenumbers

A scope "extra" appears in the second step and is gone in the third. The trailing key is twice
the segment count, so the update is recorded three times. The encoder's forward region keeps its
key and is lowered once; its backward region moves twice and is lowered three times.

## AParameterNoStepTracesKeepsAZeroGradientAndItsValue

A third layer is bound and paired with the update and never traced. After two steps its gradient
tensor holds zeros and its weight holds the bytes it started with.

## AStepWithoutBackwardRunsItsForwardNodesWhenAValueIsRead

`read_scalar` of an untouched step runs it. No update was applied, so none was recorded.

## AValueTheStepDidNotKeepCannotBeReadAfterTheRun

After `apply` the model's result is not readable: it sits inside the step's one region and was
not passed to `keep`. The loss is, because `backward` keeps it. `keep` after the run is refused.

## TracingAfterTheStepRanIsRefused

A node emitted after `finish` makes the next read return `Unsupported`.

## AStateTheSessionDoesNotHoldIsRefused, BindingAStateTwiceIsRefused

`Step::state` and `Session::gradient` refuse an identity that was never bound, and
`Session::bind_state` refuses one that already is.

## ASecondLiveStepStopsTheProcess

A session traces one step at a time.

Suites: `Medium_EagerStepCost`.

Checks that the steps of a settled eager loop are the kind a cost reading would need: unchanged
ones build and record no region. The time of an eager step is read from the `eager.*` zones of a
profiling record; `StepCost` keeps the region counts only.

This file needs an engine.

`K_WARM_STEPS`, 3, run before the measured ones so every region is recorded. `K_MEASURED_STEPS`
is 20. `K_RATE` is 1e-6 so that fifty RELU layers stay finite for the length of the run.
`widths_of(layers)` gives a model 16 wide with 4 results.

## AnUnchangedStepOfTheTwoLayerModel, AnUnchangedStepOfAFiftyLayerModel

`run_settled` runs a loop with no scope and with one scope per layer. It returns
`Incompatible` when a measured step built or recorded a region, since that step would not be an
unchanged one.

## AScopeThatAppearsAndDisappears

A fifty-layer loop with one scope per layer alternates a step with a scope "extra" and a step
without it, ten times each. The case fails when no region was recorded again, because the
steps would then not have been renumberings.
