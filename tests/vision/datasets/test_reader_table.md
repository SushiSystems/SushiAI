# tests/vision/datasets/test_reader_table.cpp


Suites: `Small_DatasetsReaderTableRows`.

Tests `reader_table` and `find_reader` in
`modules/vision/datasets/include/SushiAI/vision/datasets/reader_table.hpp`. The names of the
eight rows, the keys each row reads and what the rows pass to their readers are tested in
`test_reader_section.cpp`, under `Small_DatasetsReaderTable`. This file pins the two promises
those cases leave out: no row has a null reader, and a lookup is exact.

## EveryRowPointsAtAReader

`read` is not null in any row.

## NoTwoRowsShareAName

The names of the rows are unique.

## ALookupMatchesTheWholeNameAndItsCase

"yolo_txt" finds a row. "YOLO_TXT", "yolo", "yolo_txt " with a trailing space and the empty
name find none.
