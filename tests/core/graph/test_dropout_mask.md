# tests/core/graph/test_dropout_mask.cpp


Suites: `Small_DropoutMask`.

Tests the functions of `modules/core/graph/include/SushiAI/graph/dropout_mask.hpp`:
`splitmix64_at`, `dropout_site_seed`, `dropout_bits`, `dropout_keeps` and `fill_keep_mask`.
Every case but the first two draws from the site seed 2026. A mask is compared as a string of
'1' for a kept element and '0' for a dropped one.

## TheSequenceOfSeedZeroIsThePublishedSplitMix64

`splitmix64_at(0, i)` for i from 0 to 3 is 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
0x06C45D188009454F and 0xF88BB8A8724C81EC.

## EachSiteOfATraceHasItsOwnSeed

`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are the first two numbers of the case
above. `dropout_site_seed(7, 0)` is 0x63CBE1E459320DD7 and `dropout_site_seed(7, 1)` is
0x044C3CD7F43C661C.

## TheBitsOfASeedARunAndAnElementAreGolden

`dropout_bits` for elements 0 to 3 of runs 0 and 1 equals eight 64-bit constants written in the
test.

## TheMaskOfARunIsGolden

At a keep of 0.5 the first 32 elements of run 0 are "11110000011001111110011111110000" and those
of run 1 are "01110100100110000111111111000011".

## TheFillAgreesWithTheElementwiseQuestion

For 257 elements of run 5 at a keep of 0.7, each element of the filled mask is 1 exactly when
`dropout_keeps` returns true for it.

## AShorterFillIsAPrefixOfALongerOne

A fill of 40 elements of run 2 at a keep of 0.9 equals the first 40 elements of a fill of 96.

## TheSameInputsGiveTheSameMask

Two fills of 4096 elements of run 9 at a keep of 0.8 are equal.

## TwoRunsAndTwoSitesDrawDifferentMasks

Runs 0 and 1 at a keep of 0.5 differ in 2033 of 4096 elements. The masks of the site seeds
`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are not equal.

## TheKeptFractionIsTheKeepProbability

Over 2^20 draws the kept fraction is within 0.002 of the keep: 0.9 and 0.5 in run 0, and 0.9 in
run 3.

## TheTwoEndsOfTheKeepRangeAreExact

A keep of 1 keeps all 64 elements and a keep of 0 keeps none.
