# tests/vision/detect/test_float_order.cpp


Suites: `Small_DetectFloatOrder`.

Tests `first_non_finite` and `ranks_above` of `float_order.hpp`. The file needs no device. `K_NAN`
is a quiet NaN and `K_INF` a positive infinity.

## FirstNonFiniteOfNoValuesIsMinusOne

An empty span gives -1.

## FirstNonFiniteOfFiniteValuesIsMinusOne

Zero, -2.5, the largest, the lowest and the smallest denormal float give -1.

## FirstNonFiniteNamesANaNAndBothInfinities

A NaN at index 1, a positive infinity at index 2 and a negative infinity at index 0 are each named
by their index.

## FirstNonFiniteNamesTheFirstOfSeveral

For 0, infinity, NaN and negative infinity the index is 1.

## RanksAboveOrdersNumbersLargestFirst

2 ranks above 1 and 1 does not rank above 2. 0 ranks above -1, infinity above 1, and 1 above
negative infinity.

## RanksAboveIsFalseBothWaysForEqualNumbers

1.5 against 1.5, 0 against -0 and back, and infinity against infinity rank above in neither
direction, so the relation is strict.

## RanksAboveSendsANaNBelowEveryNumber

A NaN does not rank above 1 or negative infinity; both rank above it. A NaN against a NaN ranks
above in neither direction.

## SortingWithRanksAboveLeavesTheNaNsLast

`std::sort` with `ranks_above` over NaN, 1, NaN, 3, -1 and 2 gives 3, 2, 1, -1 and then the two
NaNs.
