# tests/layers/nn/test_leaky_relu.cpp


Suites: `Small_LeakyReLU`.

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `modules/layers/nn/include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReLUDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

## is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

## is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
the trace throws `Error`. It uses `Test::capture_error` from `modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp`,
so an exception of another type does not count as a rejection.

## TheDefaultSlopeIsOneHundredth

A default-constructed layer has slope 0.01.

## ForwardEmitsOneLeakyReLUNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a K_LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

## ForwardKeepsTheShapeAndTypeOfAFeatureBatch

A [2, 4] input traces to a [2, 4] value of the input's element type.

## ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

A [2, 8, 6, 3] input traces to a value of the same shape tagged NHWC.

## ForwardCarriesTheLayoutOfItsInput

The input is converted to NCHW first. The result is tagged NCHW and has the converted value's
shape. The activation is elementwise and reads no axis, so it accepts either layout and
passes the tag on, where the spatial layers refuse NCHW.

## OutputShapeAgreesWithForwardForAFeatureVector

`output_shape([7])` is [7], and the traced shape of a [5, 7] input is a batch of it.

## OutputShapeAgreesWithForwardForAnImage

`output_shape([8, 6, 4])` is [8, 6, 4], and the traced shape of a [3, 8, 6, 4] input is a
batch of it.

## ForwardRejectsASlopeOfZero

A slope of exactly 0 is rejected. The range is open at this end.

## ForwardRejectsASlopeOfOne

A slope of exactly 1 is rejected. The range is open at this end too.

## ForwardRejectsANegativeSlope

A slope of -0.01 is rejected.

## ForwardRejectsASlopeAboveOne

A slope of 1.5 is rejected.

## ForwardRejectsANaNSlope

A NaN slope is rejected. NaN compares false against both bounds, so a check written as "throw
when the slope is at or below 0 or at or above 1" lets it through.

## ForwardAcceptsSlopesJustInsideTheOpenRange

The smallest positive normal float32 and 1 minus the float32 epsilon are both accepted. Each
lies inside the range and still differs from its bound after narrowing to float32.

## OutputShapeRejectsASlopeOutsideTheOpenRange

For the five slopes the forward cases reject, `output_shape([4])` throws with a message
containing "LeakyReLU" and "slope". A factory calls `output_shape()` and not `forward()`, so
this is the check that refuses a slope when a model file is read.

## ForwardNamesTheLayerAndTheSlopeItRefuses

`forward()` with slope 1.5 throws with a message containing "LeakyReLU" and "slope".

## ARejectedSlopeEmitsNothing

After `forward()` throws for a slope of 1, the node count and the value count are unchanged.

## TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.
