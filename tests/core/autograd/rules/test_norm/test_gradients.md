# tests/core/autograd/rules/test_norm/test_gradients.cpp


Suites: `Medium_Autograd`.

Tests the backward rules of batch norm and channel affine against central differences in FLOAT64, through the `GradientProbe` of `tests/support/gradient_probe.hpp`. The method is described in [`tests/core/autograd/README.md`](../../README.md).

## BatchNormsGradientAccountsForTheStatisticsItComputed

The batch mean and variance are functions of x, so dx has three terms. Treating the statistics
as constants gives a gradient that is right on average over the batch and wrong per element.
The model still trains, slightly worse, and nothing points at the cause.

## BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput

The same node is differentiated at gamma. It is a separate case because the `Probe` checks one
variable. dgamma is the term that would go unnoticed if it were summed over the wrong axis,
since the result is a [C] value of the right shape either way.

## BatchNormsShiftGradientIsThePlainSumOfDy

The same batch norm over a `[4, 3, 3, 5]` input is differentiated at beta, its shift of five
elements. The logits are the spatial mean of the output through a `log_softmax`, against one-hot
targets.

## ChannelAffineDifferentiatesAtItsInput

The variable is the `[2, 4, 4, 6]` input of a `channel_affine` whose scale and shift are parameters
filled with noise. The logits are the spatial mean through a `log_softmax`, against one-hot targets.

## ChannelAffinesScaleGradientSumsOverNHAndW

dgamma is a sum over three axes at once. A sum over the wrong ones, per sample where it should
be per channel, is still a [C]-shaped value when N equals C. That is why the case is checked
numerically and not by shape.

## ChannelAffinesShiftGradientSumsOverNHAndW

The same node is differentiated at beta, its shift of six elements.
