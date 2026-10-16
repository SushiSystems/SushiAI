# tests/core/eager/test_partition.cpp


Suites: `Small_Partition`.

Tests the defaults of `Partition` and the shape a partition of a traced step has.

This file needs no device. The last case partitions a host-side step that
`eager_test_support.hpp` traces; `test_scope_partitioner.md` covers how the keys are chosen.

## TheEmptyKeyIsZero

`K_NO_REGION` equals zero.

## ADefaultRegionInfoNamesNoRegion

A default `RegionInfo` has the key `K_NO_REGION`, an empty name and the direction
`Direction::Forward`.

## ADefaultPartitionIsEmpty

A default `Partition` holds no node entry, no region, and its `optimizer_key` is `K_NO_REGION`.

## TheOptimizerKeyIsAboveEveryRegion

The step of `encoder_head_loss` with a K_TANH head is partitioned. `region_of_node` has one
entry per node of the graph, and every region key and every node's key is below
`optimizer_key`, so the optimizer update recorded under that key runs after every region.
