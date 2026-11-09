# tests/vision/datasets/test_fitted_sample.cpp


Suites: `Small_DatasetsFittedSample`.

Tests `FittedSample` in
`modules/vision/datasets/include/SushiAI/vision/datasets/fitted_sample.hpp`. The header declares
one struct and no function, so the case pins the way a tabular sample refers to its row. How a pipeline fills a sample is tested in
`test_fit_pipeline.cpp` and `test_augment_pipeline.cpp`.

## ATabularSampleViewsTheRowOfItsTable

A sample whose `features` and `targets` are the views of row 1 of a table of two features and
one target holds no pixel, and its views start at values 3 and 5 of the table. The sample
therefore depends on the table outliving it, as the header states.
