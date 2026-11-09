# tests/vision/datasets/test_sample_pipeline.cpp


Suites: `Small_SamplePipelineContract`, `Medium_SamplePipelineContract` and
`Medium_SamplePipelineThreads`.

Tests the interface `SamplePipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_pipeline.hpp`. The header
declares the interface only, so the cases are a contract that runs over each implementation
through a pointer to the base: `InMemorySamplePipeline`, which reads no file, and `FitPipeline`
and `AugmentPipeline`, which read three raw RGB images of 16 by 16 pixels written under the
system's temporary directory. The earlier fake pipeline is gone. The two cases that only reached
the fake, the epoch and the failure of an index, are deleted: `test_augment_pipeline.cpp` holds
the epoch promise and `test_fit_pipeline.cpp` the refusals. The pipelines of the module are
tested in `test_fit_pipeline.cpp` and `test_augment_pipeline.cpp`, and the table that names them
in `test_pipeline_table.cpp`.

## three_samples

`three_samples` returns a list of three files and three samples, sample `i` in file `i` with
class `i`. `three_images` gives the same list with files that are raw images on disk.

## ProduceFillsOneSamplePerIndexInOrder

`Small_SamplePipelineContract` has the row `InMemory`, and `Medium_SamplePipelineContract` has the
rows `Fit` and `Augment`. For each, the indices 2 and 0 fill the first fitted sample from sample 2
and the second from sample 0.

## TwoThreadsGetTheResultOfOne

`Medium_SamplePipelineThreads`. An `InMemorySamplePipeline` called from two threads at once with
the indices 2, 0 and 1 gives each the file and the class that one thread alone gives, as the
header documents for `produce`.
