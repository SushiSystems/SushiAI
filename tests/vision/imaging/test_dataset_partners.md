# tests/vision/imaging: test_dataset_partners.cpp, test_recent_partners.cpp


The two sources of a mosaic's three partner images.

Suites: `Small_ImagingPartners`.

Tests the two sources of a mosaic's three partner images.

This file needs no device. `Small_ImagingPartners` covers `DatasetPartners` of
`modules/vision/imaging/include/SushiAI/vision/imaging/dataset_partners.hpp`, which draws partners from the whole dataset,
and `RecentPartners` of `modules/vision/imaging/include/SushiAI/vision/imaging/recent_partners.hpp`, which draws them from a
buffer of recently seen indices.

## buffer_after

`buffer_after` passes a list of indices to `RecentPartners::choose`, one by one with a
generator seeded 1, and returns the buffer, oldest first.

## DatasetSourceReturnsThreeIndexDrawsInOrder

`choose` on a dataset of 1000 returns three values equal to three `index(1000)` draws of a copy
of the generator, in draw order, and the two generators are equal afterwards.

## DatasetSourceCoversTheDatasetAndNeverLeavesIt

Over 300 seeds and a dataset of 7, every partner is below 7 and every index is a partner at
least once.

## DatasetSourceKeepsNoStateBetweenSamples

A source that already answered two calls and a new source return the same partners for the
same index and seed.

## DatasetSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and one above the largest `uint32` throw, and the largest `uint32` is
accepted. `choose(4)` on a dataset of 4 fails with "index 4" and makes no draw.

## RecentCapacityIsTheSmallestOfItsThreeBounds

`recent_capacity` is 5 for a dataset of 5, 128 for a batch of 16 and 1000 for a batch of 200:
the smallest of the dataset size, `K_RECENT_BATCHES` (8) batches and `K_RECENT_LIMIT` (1000).

## RecentSourceGivesTheFirstSampleItselfThreeTimes

The first call, for index 42, returns 42 three times, and the buffer holds 42.

## RecentSourceEvictsTheOldestIndexAtCapacity

With capacity 3 the buffer goes {0, 1}, {1, 2}, {2, 0}. With capacity 2 it holds one index at a
time. The reference removes the oldest entry when the buffer has more than one entry and has
reached its capacity, so a full buffer holds capacity minus one indices.

## RecentSourceDoesNotAddAnIndexItAlreadyHolds

After the indices 7, 9, 7, 7, 9 the buffer is {7, 9}.

## RecentSourceOfOneImageNeverEmptiesItsBuffer

With a dataset of one image, two calls both return 0 three times and the buffer holds 0.

## RecentSourceHoldsTheLastIndicesOfALongRun

With capacity 16, after the indices 0 to 19 the buffer holds 15 indices, from 5 to 19.

## RecentSourceDrawsThreeEntriesOfItsBuffer

After 40 earlier indices and the index 500, the buffer has 41 entries. Each partner is the
buffer entry that one `index(41)` draw of a copied generator names, in draw order, and the
source makes three draws.

## RecentSourceCoversItsBuffer

With the indices 10 to 14 buffered, over 200 seeds every partner is one of them and each is
drawn at least once.

## RecentSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and batch sizes of 0 and -3 throw. `choose(4)` on a dataset of 4 throws
with "index 4", makes no draw and leaves the buffer empty.
