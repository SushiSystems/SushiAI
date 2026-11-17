# tests/core/graph/lowering/test_unlowered.cpp


Suites: `Small_Unlowered`, `Small_UnloweredPair`.

Tests the contents of `K_UNLOWERED`. That none of its operations has a lowering row is pinned by
`tests/core/graph/lowering/test_all.cpp`; this file pins which operations the list names.

## TheListNamesTheFourOperationsRecordedBesideTheGraph

The list has six entries: ADAMW_UPDATE, SGD_UPDATE, CLIP_SCALE, BATCHNORM_UPDATE_STATS, LAYER_NORM_DEVIATION and LAYER_NORM_APPLY.

## NoOperationIsListedTwice

One row per pair of positions, fifteen in all: the two entries are different operations, so the six entries are six operations.
