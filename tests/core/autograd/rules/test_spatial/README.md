# tests/core/autograd/rules/test_spatial

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of im2col, col2im, pooling, upsampling, concat, slice and depthwise convolution against central differences in FLOAT64. |
| `test_rules.cpp` | Tests the shapes the builder infers for spatial operations and the layouts it refuses. |
