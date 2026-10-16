# tests/core/data/test_materialised_dataset.cpp


Suites: `Medium_MaterialisedDataset`.

Tests `Data::MaterialisedDataset`, `Data::share_batch` and how `Data::prepare_batch` treats
tensors a dataset shares.

The fixture needs an engine. `two_batches` builds a dataset named "held" from two batches of 3
samples, each with a float32 [4] input and one `labels` role of int32 over 2 classes. The
layouts `Data::allocate_batch` and `Data::prepare_batch` give a fresh batch are pinned in
`test_target.cpp`.

## StatesWhatItWasBuiltWith

The dataset reports a batch size of 3, 2 batches, the name "held", float32 inputs of shape [4]
and the one role.

## AReadSharesTheTensorsOfItsIndex

Reads of batches 0 and 1 hand out different inputs, and two reads of batch 1 hand out the same
input and target data pointers.

## ShareBatchCopiesNoElement

`Data::share_batch` leaves the batch with the data pointers of the held batch and its role.

## PrepareBatchReplacesTensorsTheDatasetShares

After a read, the batch's inputs are also held by the dataset. `Data::prepare_batch` with the
same layout replaces them with a [3, 4] tensor at another address, so a caller that fills the
batch cannot write into the dataset.
