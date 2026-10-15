# tests/vision/imaging/test_random.cpp


Suites: `Unit_ImagingRandom`.

Tests the generator every augmentation draws from.

This file needs no device. `Unit_ImagingRandom` covers `Random` of
`modules/vision/imaging/include/SushiAI/vision/imaging/random.hpp`: a 64-bit counter advanced by `K_INCREMENT` and passed
through `mix`. The first cases follow the arithmetic by hand, with each intermediate value as a
literal, so the sequence is fixed by this file and not by a second copy of the code. The
statistical cases use one seed and state their bound as a multiple of the standard error.

## Constants

`K_SEED` is 2026, the seed of every statistical case. `K_SAMPLES` is 65536. `K_UNIFORM_VALUES`
is 2^24, the number of values `uniform()` can return.

## MixFollowsTheFinaliserStepByStep

For the value 0x9E3779B97F4A7C15 the case asserts each of the five steps of `mix`: the shift by
30 and xor, the product with `K_FIRST_MULTIPLIER`, the shift by 27 and xor, the product with
`K_SECOND_MULTIPLIER` and the shift by 31 and xor. `mix` returns the last literal.

## MixOfTheSecondCounterFollowsTheFinaliserStepByStep

The same five steps for 0x3C6EF372FE94F82A, the second counter of seed 0.

## SeedZeroReproducesTheHandComputedSequence

The first four `next_u64` values of seed 0 are the literals the case lists, and the counter
after draw n is n times `K_INCREMENT` modulo 2^64.

## TheCounterStartsAtTheSeed

`state()` of a new generator is its seed. The first draw of seed 42 is
`mix(42 + K_INCREMENT)`, and the counter is then `42 + K_INCREMENT`.

## NextU32IsTheHighHalfOfOneDraw

The first two `next_u32` values of seed 0 are the high 32 bits of its first two draws, and two
calls advance the counter twice.

## UniformIsTheTop24BitsScaledByTwoToTheMinus24

The first three `uniform()` values of seed 0 are 14819496, 7239838 and 443485 over 2^24, the
top 24 bits of each draw.

## UniformStaysInTheHalfOpenUnitInterval

65536 draws are all at or above 0 and below 1.

## UniformHasTheMeanAndVarianceOfAUniformVariable

Over 65536 draws the mean is 0.5 within 0.006 and the variance 1/12 within 0.0015, five
standard errors each.

## RangedUniformMapsTheUnitDrawLinearly

`uniform(-1, 1)` and `uniform(0.5, 1.5)` of seed 0 return `low + (high - low) * u` for the
first two unit draws, as exact floats, and make one draw each.

## RangedUniformStaysInItsHalfOpenInterval

65536 draws of `uniform(0.4, 0.6)` are all at or above 0.4 and below 0.6.

## RangedUniformNeverReturnsItsUpperBound

With `high` the float after `low`, 1024 draws all return `low`. A sum that rounds up to `high`
is brought back.

## RangedUniformOverAnEmptyRangeStillDraws

`uniform(0.25, 0.25)` returns 0.25 and advances the generator by one draw.

## IndexIsTheHighWordOfTheProduct

`index(10)` of seed 0 returns 8, 4 and 0: the high 32 bits of each 32-bit draw times 10.

## IndexOfOneIsAlwaysZeroAndStillDraws

64 calls of `index(1)` return 0 and advance the generator 64 times.

## IndexStaysBelowTheCountAndFillsEveryBucketEvenly

70000 draws of `index(7)` are all below 7, and each bucket holds 10000 within 500, above five
standard deviations of 93. `K_BUCKETS` and `K_DRAWS` are local to the case.

## IndexReachesTheLargestCount

`index(0xFFFFFFFF)` of seed 0 returns 0xE220A838.

## BernoulliComparesOneUniformDrawWithTheProbability

`bernoulli(0.5)` of seed 0 returns false, true, true for the unit draws 0.8833, 0.4315 and
0.0264, one draw per call.

## BernoulliNeverSucceedsAtZeroAndAlwaysSucceedsAtOne

Over 65536 draws, probability 0 never succeeds and probability 1 always does, and both make
one draw per call.

## AZeroDrawIsTheLowerBoundAndFailsAZeroProbability

`mix(0)` is 0, so a seed of minus `K_INCREMENT` makes the first draw 64 zero bits. `uniform()`
returns 0, `bernoulli(0)` returns false and `uniform(0.4, 0.6)` returns 0.4. A draw of exactly
0 does not pass a probability of 0.

## BernoulliSucceedsAtTheRateOfItsProbability

Over 65536 draws `bernoulli(0.25)` succeeds at a rate of 0.25 within 0.01.

## DeriveStartsAtTheMixOfTheStateAndTheKeyedMix

`Random(0).derive(0).state()` is 0x48218226FF3CD4BF. The case follows the five steps of `mix`
from the keyed value 0xE220A8397B1DCDAF.

## DeriveDependsOnTheKeyAndOnTheParentState

`Random(0).derive(1)` and `Random(7).derive(5)` have the two states the case lists. Two keys
of one parent give different generators, and so does one key of two parents.

## DeriveLeavesTheParentUntouched

After `derive(3)` the parent's state is its seed, the child differs from the parent, and a
second `derive(3)` equals the first.

## ACopyReplaysTheSequenceFromThePointOfTheCopy

A copy taken after one draw equals the original, and their next 16 draws are equal.

## EqualityComparesTheCounter

Two generators of one seed are equal, and are unequal after one of them draws.
