# tests/core/eager/test_partition.cpp


Suites: `Small_Partition`.

Tests the shape a partition of a traced step has.

This file needs no device. Every case partitions a host-side step that
`eager_test_support.hpp` traces; `test_scope_partitioner.md` covers how the keys are chosen.

## ThePartitionOfATracedStepCoversEveryNode

The step of `encoder_head_loss` with a K_TANH head is partitioned. `region_of_node` has one
entry per node of the graph and `regions` is not empty.

## TheOptimizerKeyIsAboveEveryRegionKey

Every region key of that partition is below `optimizer_key`, so the optimizer update recorded
under that key runs after every region.

## TheOptimizerKeyIsAboveEveryNodeKey

Every entry of `region_of_node` is below `optimizer_key`.
