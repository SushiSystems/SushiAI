# tests/core/graph/test_tensor_source.cpp


Suites: `Small_InMemoryTensorSource`.

Tests `Graph::InMemoryTensorSource`, the adapter of the `Graph::TensorSource` port in
`modules/core/graph/include/SushiAI/graph/tensor_source.hpp`. The file needs no device. How a
`TensorPool` uses a source is tested in `test_tensor_pool.cpp`.

## ATensorHoldsTheExtentsAndTheTypeItWasAskedFor

`create` of a float64 [2, 5] returns a tensor of rank 2 with the extents 2 and 5, the type
float64 and 10 elements.

## ATensorHoldsNoStorage

A tensor the source returns has no storage.

## EveryRequestIsKeptInOrder

After `create` of a float32 [3] and an int32 [2, 2], `requests()` holds those two shapes and types,
in that order.
