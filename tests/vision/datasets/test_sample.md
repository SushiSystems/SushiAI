# tests/vision/datasets/test_sample.cpp


Suites: `Small_DatasetsSample`.

Tests `Sample` and `SampleList` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample.hpp`. The header declares two
structs and no function, so the cases pin the defaults a reader starts from and the indexing
into `SampleList::files`. What each reader puts in a list is tested in the reader's own file,
such as `test_yolo_txt.cpp` and `test_mask_png.cpp`.

## ADefaultSampleCarriesNoLabel

A default `Sample` names file 0 and has no `stored`, no `class_index`, no objects and no
`mask`, so a reader sets only what its layout holds.

## ADefaultListIsEmpty

A default `SampleList` has no file, no sample, no class name, a `class_count` of 0 and a table
of no row.

## SamplesReferToTheFilesOfTheirListByIndex

A list of the files "images/a.png" and "masks/a.png" holds one sample whose `file` is 0 and
whose `mask` is 1. Indexing `files` with them gives the two paths.
