# tests/core/autograd/rules/test_linalg

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rules of matmul, bias add, sum rows, gather, scatter, take and put against central differences in FLOAT64. |
| `test_rules.cpp` | Tests the backward rules of the linear-algebra table: the causal mask and the row gather. |
