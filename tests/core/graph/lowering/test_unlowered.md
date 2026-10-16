# tests/core/graph/lowering/test_unlowered.cpp


Suites: `Small_Unlowered`.

Tests the contents of `K_UNLOWERED`. That none of its operations has a lowering row is pinned by
`tests/core/graph/lowering/test_all.cpp`; this file pins which operations the list names.

## TheListNamesTheFourOperationsRecordedBesideTheGraph

The list holds ADAMW_UPDATE, SGD_UPDATE, CLIP_SCALE and BATCHNORM_UPDATE_STATS.

## NoOperationIsListedTwice

No two entries are equal, so the four entries are four operations.
