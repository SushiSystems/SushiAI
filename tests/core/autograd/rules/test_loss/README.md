# tests/core/autograd/rules/test_loss

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_gradients.cpp` | Tests the backward rule of the fused log-softmax and negative log-likelihood against central differences in FLOAT64. |
| `test_rules.cpp` | Tests the loss rule table and which rule starts from the implicit scalar seed. |
