# tests/vision/datasets/test_task_table.cpp


Suites: `Small_DatasetsTaskTableRows`.

Tests `task_table` and `find_task` in
`modules/vision/datasets/include/SushiAI/vision/datasets/task_table.hpp`. The names of the six
rows, the fit and the keys of each, and the collator each row builds are tested in
`test_reader_section.cpp`, under `Small_DatasetsTaskTable`. This file pins the two promises
those cases leave out: no row has a null function, and a lookup is exact.

## EveryRowPointsAtAFactoryAndABuilder

`make_collator` and `build` are not null in any row.

## NoTwoRowsShareAName

The names of the rows are unique.

## ALookupMatchesTheWholeNameAndItsCase

"detection" finds a row. "Detection", "detect", "detection " with a trailing space and the
empty name find none.
