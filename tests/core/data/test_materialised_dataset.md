# tests/core/data/test_materialised_dataset.cpp


Suites: `Medium_MaterialisedDataset`.

Tests `Data::MaterialisedDataset` and `Data::share_batch`.

The fixture needs an engine. `two_batches` builds a dataset named "held" from two batches of 3
samples, each with a float32 [4] input and one `labels` role of int32 over 2 classes. The layouts
`Data::allocate_batch` and `Data::prepare_batch` give a fresh batch are pinned in `test_target.cpp`.
The reads of a materialised set that every dataset shares are pinned by the contract suites in
`test_dataset.cpp`.

## AReadGivesTensorsThatMatchWhatTheDatasetStates

A read of batch 1 gives inputs of the type the dataset states, with the batch size as the first
extent and the sample shape as the second. The one target has the stated role and type and one
value per sample.

## ShareBatchCopiesNoElement

`Data::share_batch` leaves the batch with the data pointers of the held batch and its role.

## ShareBatchGivesTheRolesAndTheLayoutOfTheHeldBatch

Over a batch an `InMemoryBatchAllocator` described, `share_batch` gives a batch with the same
rank, 3 samples on axis 0 and the one role `labels` as an int32 target. The case needs no device;
`ShareBatchCopiesNoElement` keeps the claim that no element is copied, since only a tensor with
storage can show it.
