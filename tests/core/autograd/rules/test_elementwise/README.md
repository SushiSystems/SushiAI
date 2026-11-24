# tests/core/autograd/rules/test_elementwise

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of add, sub, mul, scale, exp, accumulate, sqrt, log and pow against central differences in FLOAT64. |
| `test_rules.cpp` | Tests that the elementwise rule table pairs each operation with its own rule. |
