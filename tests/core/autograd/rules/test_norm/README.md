# tests/core/autograd/rules/test_norm

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of batch norm and channel affine against central differences in FLOAT64. |
| `test_rules.cpp` | Tests the IR the backward rule of LAYER_NORM appends, and which operands it reaches. |
