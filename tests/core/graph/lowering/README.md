# tests/core/graph/lowering

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_activation.cpp` | Tests the activation lowering table and the gradient rows that read a forward output. |
| `test_all.cpp` | Tests that each identity of the operation table has one lowering row or is unlowered. |
| `test_broadcast.cpp` | Tests the broadcast lowering table, the four broadcast operations and the sum along. |
| `test_context.cpp` | Tests what a lowering context gives a row: its node, its tensors, counts and lists. |
| `test_elementwise.cpp` | Tests the elementwise lowering table, the run-time factor of DYNAMIC_SCALE and CAST. |
| `test_fused.cpp` | Tests the fused lowering table and the GEMM row with a bias and a ReLU epilogue. |
| `test_loss.cpp` | Tests the loss lowerings: log-softmax and cross-entropy stay finite and exact far below the batch maximum. |
| `test_mask.cpp` | Tests the causal mask lowering on a device: a HALF mask holds the lowest finite value. |
| `test_matmul.cpp` | Tests the matmul, bias add and sum rows lowerings on a device, with the softmax and the product's gradient of the attention operations. |
| `test_norm.cpp` | Tests the normalization lowering table and the scratch buffers of batch normalization. |
| `test_random.cpp` | Tests the shape rule and the emitted IR of DROPOUT, its row of the lowering table, and the masks its lowered step draws on an engine. |
| `test_row_index.cpp` | Tests the gather, scatter, take along, put along and argmax lowerings on a device. |
| `test_rule.cpp` | Tests the lowering row type and the compile-time join of two row tables. |
| `test_shape.cpp` | Tests the shape lowering table. |
| `test_spatial.cpp` | Tests the spatial lowering table and the gradient rows of upsampling and the spatial mean. |
| `test_unlowered.cpp` | Tests the list of operations that are recorded beside a graph and have no row. |
