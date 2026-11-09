# tests/layers/nn/test_mish.cpp


Suites: `Small_Mish`, `Medium_MishValues`.

Tests `NN::Mish` in `modules/layers/nn/include/SushiAI/nn/mish.hpp`, which has no parameter and
no attribute.

Three `static_assert` lines at the top of the file hold that `Mish` is a `Module`, is not a
`ModuleContainer`, and that `Mish::fields()` is an empty tuple.

## ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

## ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::NHWC`.

## ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

## OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

## TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## Medium_MishValues.SweepMatchesTheExactForm

The layer runs over a FLOAT32 row from -30 to 30 in steps of 0.5 on the engine. Each result is
within `K_RELATIVE_TOLERANCE` (32 float unit roundoffs) of x * tanh(log1p(exp(x))) computed in
double; the sweep holds the points -20, -1, 0, 1 and 20. The case is medium because the
kernels run on the device and have no host evaluation.
