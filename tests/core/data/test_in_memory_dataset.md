# tests/core/data/test_in_memory_dataset.cpp


Suites: `Small_InMemoryDataset`, `Small_InMemoryEpochDataset`, `Medium_InMemoryDatasetDeathTest`.

Tests `Data::InMemoryDataset` and `Data::InMemoryEpochDataset` in
`modules/core/data/include/SushiAI/data/in_memory_dataset.hpp`. The file needs no device. Every
dataset has batches of 4 samples, an input of three float32 values per sample and one int32 role,
`labels`, over 2 classes. A batch the default maker builds holds no storage.

## StatesTheExtentsItWasBuiltWith

A dataset of 5 batches states 5 batches, 4 samples per batch, the sample shape [3], float32 and
the one role `labels`.

## AReadGivesTheLayoutTheDatasetStates

`read(1, batch)` fills a [4, 3] input and one `labels` target of type int32.

## TheJournalHoldsTheIndicesReadInOrder

After reads of 3, 0 and 3, `journal()` holds 3, 0, 3.

## AReadOfTheFailingIndexIsRefused

After `fail_at(2)`, `read(2)` fails with `InvalidFile`, "batch 2" and "unreadable"; `read(1)`
succeeds; the journal holds 2 and 1.

## ABadRangeIsRefusedWithAStatusAndReadsNothing

`read_rows(0, {3, 2})` fails with `InvalidDescription` naming "Rows 3 to 5", and the journal stays
empty.

## ABatchMakerBuildsTheBatchOfAnIndex

With a maker that names the target of a batch "batch-<index>", `read(2)` hands out a batch whose
target role is "batch-2".

## AReadPastTheLastBatchStopsTheProcess

`read(2)` on a dataset of 2 batches stops the process with a report that holds "Batch 2 is out
of range". It is a death test.

## OnlyTheEpochDatasetIsEpochAware

A `dynamic_cast` to `Data::EpochAware` fails for an `InMemoryDataset` and succeeds for an
`InMemoryEpochDataset`.

## TheBatchMakerIsToldTheEpochLastBegun

A maker that names the target of a batch by its epoch gives "0" before any epoch begins, and "6"
after `Data::begin_epoch(dataset, 6)`.
