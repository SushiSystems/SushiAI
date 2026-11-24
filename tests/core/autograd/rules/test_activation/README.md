# tests/core/autograd/rules/test_activation

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of relu, leaky relu, gelu, sigmoid and tanh against central differences in FLOAT64. |
| `test_rules.cpp` | Tests that the activation rule table sends each activation to activation_backward. |
