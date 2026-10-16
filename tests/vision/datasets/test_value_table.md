# tests/vision/datasets/test_value_table.cpp


Suites: `Small_DatasetsValueTable`.

Tests `ValueTable` in `modules/vision/datasets/include/SushiAI/vision/datasets/value_table.hpp`:
`row_width`, `row_count` and the two row views. The cases build tables in memory. A view copies
nothing, so its values are compared exactly. How the `csv` reader fills a table is tested in
`test_csv.cpp`.

## three_rows

`three_rows` returns a table of the columns "x", "y" and "label", two features and one target,
holding the rows 1, 2, 10, then 3, 4, 20, then 5, 6, 30.

## ADefaultTableHoldsNoRow

A default table, which is what a list of images holds, has a `row_width` of 0 and a `row_count`
of 0, so the division in `row_count` never runs with a zero width.

## TheRowWidthIsTheFeaturesPlusTheTargets

`three_rows` has a `row_width` of 3 and a `row_count` of 3.

## APartialRowIsNotCounted

One value appended to `three_rows` leaves `row_count` at 3.

## TheViewsOfARowSplitItAtTheFeatureCount

`features(1)` holds 3 and 4 and `targets(1)` holds 20. The views of row 2 start at value 6 and
value 8 of `values`, so they point into the table and do not copy it.

## ATableWithoutFeaturesViewsOnlyTargets

A table of two targets and no feature over four values has two rows. `features(1)` is empty
and `targets(1)` holds 3 and 4.
