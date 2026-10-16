# tests/vision/datasets/test_sample_pipeline.cpp


Suites: `Small_DatasetsSamplePipeline`.

Tests the interface `SamplePipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_pipeline.hpp`. The header
declares the interface only, so the cases drive a fake pipeline through a pointer to the base.
The pipelines of the module are tested in `test_fit_pipeline.cpp` and
`test_augment_pipeline.cpp`, and the table that names them in `test_pipeline_table.cpp`.

## EpochPipeline

`EpochPipeline` is the fake. Its name is "fake_epoch". For each index it copies the sample's
file and sets the class to the sample's class plus the epoch, and it refuses an index the list
does not hold.

## three_samples

`three_samples` returns a list of three files and three samples, sample `i` in file `i` with
class `i`.

## ProduceFillsOneSamplePerIndexInOrder

The indices 2 and 0 fill the first fitted sample from sample 2 and the second from sample 0.
`name` read through the base is "fake_epoch".

## TheEpochReachesThePipeline

Index 1 at epoch 3 gives class 4.

## AFailureReachesTheCaller

The indices 0 and 7 over a list of three samples fail with InvalidDescription naming "sample 7".
