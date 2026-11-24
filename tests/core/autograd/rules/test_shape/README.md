# tests/core/autograd/rules/test_shape

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of reshape, transpose and layout convert against central differences in FLOAT64. |
| `test_rules.cpp` | Tests that the shape rule table pairs reshape and layout conversion with their rules. |
