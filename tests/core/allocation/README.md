# tests/core/allocation

The tests of the module `core/allocation`, one file per header of
`modules/core/allocation/include/`. The suite prefix of a case gives its ctest label.


| File | Tests |
| --- | --- |
| `test_allocation.cpp` | Tests that a tensor and an arena get the residency their `TensorAccess` names ([page](test_allocation.md)). |
| `test_host_copy.cpp` | Tests the copies between the host and a tensor of either residency ([page](test_host_copy.md)). |
