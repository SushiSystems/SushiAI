# tests/layers/nn/test_silu.cpp


Suites: `Small_SiLU`, `Medium_SiLUValues`.

Tests the SiLU layer's trace, shape inference and empty state.

The trace cases need no device; the value case runs on the engine. `NN::SiLU` in `modules/layers/nn/include/SushiAI/nn/silu.hpp` has no parameter and no
attribute.

Three `static_assert` lines at the top of the file hold that `SiLU` is a `Module`, is not a
`ModuleContainer`, and that `SiLU::fields()` is an empty tuple.

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

## Medium_SiLUValues.MatchesTheExactFormAtTheAcceptancePoints

The layer runs over a FLOAT32 row of -20, -1, 0, 1 and 20 on the engine. Each result is within
`K_RELATIVE_TOLERANCE` (32 float unit roundoffs) of x / (1 + exp(-x)) computed in double. The
case is medium because the SILU kernel runs on the device and has no host evaluation.
