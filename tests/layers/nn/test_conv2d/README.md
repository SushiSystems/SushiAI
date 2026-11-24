# tests/layers/nn/test_conv2d

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_groups.cpp` | Tests the groups of the conv2d layer: grouped and depthwise filters, their values and gradients. |
| `test_layer.cpp` | Tests the 2-D convolution layer's trace, parameters, shape inference and rejections. |
