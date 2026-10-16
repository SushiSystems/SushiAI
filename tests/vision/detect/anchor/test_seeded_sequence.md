# tests/vision/detect/anchor/test_seeded_sequence.cpp


Suites: `Small_DetectSeededSequence`.

Tests that `SeededSequence` repeats for one seed, differs between seeds and takes one number per
unit draw.

This file needs no device. `tests/vision/detect/anchor/test_anchor_clustering.cpp` compares the
first eight numbers of seed 0 with the published SplitMix64 sequence and checks the bits of one
unit draw. This file checks the properties the anchor clustering relies on across a run.

## K_SEED and K_DRAWS

`K_SEED` is the seed every case starts from. `K_DRAWS` is the 1000 draws a case compares.

## TwoSequencesOfOneSeedYieldTheSameNumbers

Two sequences built from `K_SEED` return the same 1000 numbers in the same order.

## NeighbouringSeedsStartWithDifferentNumbers

The first numbers of `K_SEED` and `K_SEED + 1` differ. SplitMix64 mixes a counter through a
bijection, so two different counters cannot give the same first number.

## AUnitDrawTakesOneNumber

After one `next_unit()` on one sequence and one `next()` on another of the same seed, the next
numbers of both are equal. A unit draw that consumed two numbers, or none, would fail it.

## EveryUnitDrawLiesInTheHalfOpenUnitInterval

1000 unit draws all lie in [0, 1).
