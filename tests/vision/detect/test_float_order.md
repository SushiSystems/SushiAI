# tests/vision/detect/test_float_order.cpp


Suites: `Small_DetectFloatOrderScan`, `Small_DetectFloatOrder` and `Small_DetectFloatOrderRank`.

Tests `first_non_finite` and `ranks_above` of `float_order.hpp`. The file needs no device. `K_NAN`
is a quiet NaN and `K_INF` a positive infinity. The scan and the relation each run over rows, and
each row's name states its input.

## NamesTheIndexOfTheFirstBadValue

One promise over five rows: the scan gives the index of the first NaN or infinity, or -1 when
there is none.

- `NoValuesGivesMinusOne`: an empty span gives -1.
- `FiniteExtremesGiveMinusOne`: zero, -2.5, the largest, the lowest and the smallest denormal
  float give -1.
- `ANaNIsNamed`, `APositiveInfinityIsNamed` and `ANegativeInfinityIsNamed`: a NaN at index 1, a
  positive infinity at index 2 and a negative infinity at index 0 are each named by their index.

## FirstNonFiniteNamesTheFirstOfSeveral

For 0, infinity, NaN and negative infinity the index is 1.

## AnswersAsTheRowStates

One promise over fourteen rows of `ranks_above`:

- Numbers: 2 ranks above 1 and 1 does not rank above 2. 0 ranks above -1, infinity above 1, and 1
  above negative infinity.
- Equal numbers: 1.5 against 1.5, 0 against -0 and back, and infinity against infinity rank above
  in neither direction, so the relation is strict.
- NaN: a NaN does not rank above 1 or negative infinity; both rank above it. A NaN against a NaN
  ranks above in neither direction.

## SortingWithRanksAboveLeavesTheNaNsLast

`std::sort` with `ranks_above` over NaN, 1, NaN, 3, -1 and 2 gives 3, 2, 1, -1 and then the two
NaNs.
