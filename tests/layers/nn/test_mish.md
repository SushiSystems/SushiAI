# tests/layers/nn/test_mish.cpp


Suites: `Unit_Mish`.

Tests `NN::Mish` in `modules/layers/nn/include/SushiAI/nn/mish.hpp`, which has no parameter and
no attribute.

Three `static_assert` lines at the top of the file hold that `Mish` is a `Module`, is not a
`ModuleContainer`, and that `Mish::fields()` is an empty tuple.

## ForwardEmitsSoftplusTanhThenMulReadingTheInput

`forward` adds three nodes in the order SOFTPLUS, TANH, MUL. SOFTPLUS reads the input, TANH
reads the SOFTPLUS result, and MUL reads the input and then the TANH result and writes the
value `forward` returns.

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
