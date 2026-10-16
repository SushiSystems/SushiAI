# tests/core/data/test_epoch.cpp


Suites: `Small_DataEpoch`.

Tests that `Data::begin_epoch` reaches a dataset that implements `EpochAware`.

This file needs no device. `EpochRecorder` is a dataset of one empty batch that also implements
`EpochAware` and records the epoch it was last told and how many calls it saw. A dataset that
does not implement `EpochAware` is covered by `Small_Epoch` in `test_threaded_loader.cpp`.

## BeginEpochReachesAnEpochAwareDataset

`Data::begin_epoch` with epoch 4, called through a `Dataset` reference, leaves the recorder at
epoch 4 after one call.

## EveryCallReachesTheDataset

After the epochs 0, 1 and 0 the recorder holds epoch 0 and counts three calls.
