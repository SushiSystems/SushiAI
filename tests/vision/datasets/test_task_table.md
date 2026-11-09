# tests/vision/datasets/test_task_table.cpp


Suites: `Small_DatasetsTaskTableRows` and `Small_DatasetsTaskTableLookup`.

Tests `task_table` and `find_task` in
`modules/vision/datasets/include/SushiAI/vision/datasets/task_table.hpp`. The fit and the keys
of each row, and the collator each row builds, are tested in `test_reader_section.cpp`, under
`Small_DatasetsTaskTableStates` and `Small_DatasetsTaskTableCollators`. This file pins the
promises those cases leave out: the names of the rows, that no row has a null function, and
that a lookup is exact.

## EveryRowPointsAtAFactoryAndABuilder

One test per row, named after its task: `make_collator` and `build` are not null.

## NoTwoRowsShareAName

The names of the rows are unique.

## ALookupMatchesTheWholeNameAndItsCase

"detection" finds a row. "Detection", "detect", "detection " with a trailing space and the
empty name find none.

## TheTableListsTheSixTasksInOrder

`task_table` lists detection, classification, binary, multi_label, regression and
segmentation, in that order.
