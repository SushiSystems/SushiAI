# tests/layers/train/test_kept_rows.cpp

Suites: `Small_KeptRows`.

Tests `KeptRows` of `kept_rows.hpp`, which rows of a batch a row mask keeps. The file needs no
device and no engine.

## TheDefaultKeepsEveryRowWhateverItsIndex

The default selection keeps rows 0, 7 and 100000, states no rows (`size()` 0) and counts none
(`count()` 0).

## AFlagListKeepsExactlyTheRowsMarkedOne

The flags 1, 0, 1, 1 keep rows 0, 2 and 3, drop row 1, state 4 rows and count 3 kept.

## AnyFlagOtherThanZeroKeepsItsRow

The flags 0, 7, 255 keep rows 1 and 2.

## ASelectionOfZeroFlagsKeepsNoRow

Three flags of 0 keep no row, state 3 rows and count 0.
