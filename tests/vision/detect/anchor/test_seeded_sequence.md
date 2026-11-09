# tests/vision/detect/anchor/test_seeded_sequence.cpp


Suites: `Small_DetectSeededSequence`.

Tests that `SeededSequence` follows the published SplitMix64 sequence, repeats for one seed,
differs between seeds and takes one number per unit draw.

This file needs no device. It is the one place that reaches `SeededSequence` in the `Detail`
namespace: the class has no public home, so its own unit test names it. The anchor clustering,
review and size tests draw their jitter from `reference_splitmix64` in `support/splitmix64.hpp`
and do not name it. The helpers `draw_numbers`, `reference_numbers` and `draw_units` collect the
draws a case compares, so no case loops over them.

## K_SEED, K_NUMBERS and K_UNIT_DRAWS

`K_SEED` is the seed the cases start from. `K_NUMBERS` is the 8 numbers a case compares and
`K_UNIT_DRAWS` the 1000 unit draws of the interval case.

## SeedZeroMatchesThePublishedNumbers

The first eight numbers of `SeededSequence(0)` equal the eight published SplitMix64 numbers the
case lists, from 0xE220A8397B1DCDAF on.

## AnotherSeedMatchesTheReferenceSequence

The first eight numbers of `K_SEED` equal `reference_splitmix64(K_SEED, index)` for the same
indices.

## TwoSequencesOfOneSeedYieldTheSameNumbers

Two sequences built from `K_SEED` return the same eight numbers in the same order.

## NeighbouringSeedsStartWithDifferentNumbers

The first numbers of `K_SEED` and `K_SEED + 1` differ. SplitMix64 mixes a counter through a
bijection, so two different counters cannot give the same first number.

## AUnitDrawTakesOneNumber

After one `next_unit()` on one sequence and one `next()` on another of the same seed, the next
numbers of both are equal. A unit draw that consumed two numbers, or none, would fail it.

## AUnitDrawIsTheUpperBitsOfTheNextNumber

The first `next_unit()` of seed 0 equals the upper 53 bits of the first reference number
divided by 2^53. The quotient is exact in double, so the comparison is an equality.

## EveryUnitDrawLiesInTheHalfOpenUnitInterval

1000 unit draws all lie in [0, 1).
