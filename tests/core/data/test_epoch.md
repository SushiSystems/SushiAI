# tests/core/data/test_epoch.cpp


Suites: `Small_DataEpochSequence`, `Small_DataEpoch`.

Tests that `Data::begin_epoch` reaches a dataset that implements `EpochAware` and leaves one that
does not.

This file needs no device. `CountingDataset` is a dataset of one empty batch that counts the reads
it serves. `EpochAwareDataset` adds `Data::InMemoryEpochAware` to it, which holds the epoch last
begun and counts the calls.

## EveryCallReachesAnEpochAwareDataset

Rows: the epoch sequences 4, and 0, 1, 0. After each sequence the dataset holds the last epoch of
it and counts one call per epoch.

## ADatasetThatIsNotEpochAwareIsLeftAlone

`Data::begin_epoch` with epoch 4 on a `CountingDataset` leaves its read count at 0.
