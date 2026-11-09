# tests/layers/train/test_trainer/test_loss_scaling.cpp


Suites: `Medium_LossScaling`, `Medium_LossScalingOverflow` and `Medium_LossScalingDirection`.

This file tests loss scaling through the trainer and the optimizer: what an fp16 overflow leaves
behind, that the scale is inert under fp32, what a skipped step leaves untouched, and that a moving
scale never recompiles the step. That an overflow is detected and halves the scale, and that every
reachable scale has an exact reciprocal, are `Small_LossScaling`'s cases in
`tests/core/optim/test_loss_scaler.cpp`.

## TheRunRecoversFromTheOverflow

The mixed-precision policy runs four epochs of four batches at a scale of 2^20, high enough that
the fp16 backward pass overflows; the growth interval is out of reach. That some step is thrown
away is the precondition. Not every step is: the backed-off scale reaches the graph, so the run
recovers. The final scale is under the starting one and the last loss is finite.

## ASkippedStepDoesNotAdvanceTheOptimizerCounter

On the same run the optimizer's step count equals the 16 steps minus the skipped ones.

## TheSameScaleInFloat32OverflowsNothing

Under the fp32 policy the same scale never skips, never backs off and never moves, and each of the
8 steps counts.

## ASkippedStepLeavesTheParametersAndMomentsBitIdentical

AdamW with a scaler and a non-zero weight decay takes one clean step, so the moments are non-zero.
A second gradient holds two infinities and a NaN. The step is skipped, the scale halves to 512, the
step count stays 1, and the parameter and both moments keep their bytes. A skip that still decayed,
or that multiplied a zero moment by an infinity, would move them.

## ACleanStepUnderTheSameScalerStillMovesEverything

The same scaler on finite gradients: the second step is not skipped, counts as step 2, and moves
the parameter and the first moment.

## TheStepIsCompiledOnceThoughTheScaleChanges

Two rows. `Growing` trains under fp32 from a scale of 1024 with a growth interval of 2, and the
scale grows. `BackingOff` trains under the mixed policy from 2^20 and the scale backs off. In both,
the scale moved (the precondition) and the trainer compiled once.

## TheSeparateUpdateKernelRefusesAScaler

`AdamW::set_loss_scaler` on an optimizer with the separate update kernel is refused as an invalid
description that names `UpdateKernel::Fused`, the setting the user must change.
