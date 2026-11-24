# tests/core/autograd/rules

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

Subfolders: `test_activation`, `test_elementwise`, `test_linalg`, `test_loss`, `test_norm`, `test_shape`, `test_spatial`.

| File | Tests |
| --- | --- |
| `compile_elementwise.cpp` | Checks at compile time that the elementwise rule table names each operation once. |
| `rule_gradient_support.hpp` | Declares the fixture and the sample points the gradient tests of the backward rules share. |
| `test_all.cpp` | Tests the combined backward rule table: its coverage, its rows and the operations that have no rule. |
| `test_broadcast.cpp` | Tests the backward rules of the four broadcast operations and the sum along against central differences in FLOAT64. |
| `test_random.cpp` | Tests the IR the backward rule of DROPOUT appends. |
