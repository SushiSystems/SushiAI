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
above. `dropout_site_seed(7, 0)` and `dropout_site_seed(7, 1)` equal the first two numbers of
the SplitMix64 sequence of seed 7, which the test computes from the published constants.

## TheBitsOfASeedARunAndAnElementFollowTheRule

`dropout_bits` for elements 0 and 3 of runs 0 and 1 equals the test's own SplitMix64 of the
SplitMix64 of the run: the sequence of the run seeds the sequence of the element.

## TheMaskOfARunKeepsTheElementsWhoseBitsFallBelowTheKeep

At a keep of 0.5 the first 32 elements of runs 0 and 1 equal the mask the rule gives: an element
is kept when its top 53 bits, as a number in [0, 1), fall below the keep.

## TheFillWritesTheMaskTheRuleGives

For 257 elements of run 5 at a keep of 0.7, the filled mask equals the mask the rule gives.

## AShorterFillIsAPrefixOfALongerOne

A fill of 40 elements of run 2 at a keep of 0.9 equals the first 40 elements of a fill of 96.

## TheSameInputsGiveTheSameMask

Two fills of 4096 elements of run 9 at a keep of 0.8 are equal.

## TwoRunsDrawDifferentMasks

Runs 0 and 1 at a keep of 0.5 draw different masks of 4096 elements.

## TwoSitesDrawDifferentMasks

The masks of the site seeds `dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are not equal.

## TheKeptFractionIsTheKeepProbability

Over 2^20 draws the kept fraction is within `K_FRACTION_TOLERANCE` of the keep: 0.9 and 0.5 in
run 0, and 0.9 in run 3. The bound is 4.1 standard deviations of the worst case, 0.5 / 2^10, which
is 0.002002 and a false-failure rate of 4.1e-5 for a fair draw. The seed is fixed.

## TheTwoEndsOfTheKeepRangeAreExact

A keep of 1 keeps all 64 elements and a keep of 0 keeps none.
