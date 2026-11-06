# tests/core/graph/lowering/test_unlowered.cpp


Suites: `Small_Unlowered`, `Small_UnloweredPair`.

Tests the contents of `K_UNLOWERED`. That none of its operations has a lowering row is pinned by
`tests/core/graph/lowering/test_all.cpp`; this file pins which operations the list names.

## TheListNamesTheFourOperationsRecordedBesideTheGraph

The list has four entries: ADAMW_UPDATE, SGD_UPDATE, CLIP_SCALE and BATCHNORM_UPDATE_STATS.

## NoOperationIsListedTwice

One row per pair of positions, six in all: the two entries are different operations, so the four entries are four operations.
