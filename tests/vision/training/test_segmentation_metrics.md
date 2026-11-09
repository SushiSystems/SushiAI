# tests/vision/training/test_segmentation_metrics.cpp


Suites: `Small_VisionTrainingSegmentationMetrics` and `Medium_VisionTrainingSegmentationMetrics`.

Tests `measure_segmentation` of `segmentation_metrics.hpp`: the mean IoU and the pixel accuracy a
per-pixel objective reports of a step. The extents are one 2 by 2 map of three classes with the
ignore index 255.

The `Small_` cases hand the function tensors from an `InMemoryTensorSource`. Such a tensor holds
its extents and element type and no element, and each refusal below is decided on those facts
before any element is read. The `Medium_` cases allocate tensors on the CPU SYCL device through
`EngineTest`, because they read the mask and the logits.

### TheMetricsEqualTheHandCount

The logits pick the classes 0, 1, 0, 1 against the mask 0, 2, ignored, 1, so two of three pixels
are right. Class 0 has an IoU of 1, class 1 of 1 / 2 and class 2 of 0, so the mean IoU is 0.5 and
the pixel accuracy 2 / 3, each to 16 double unit roundoffs. Two metrics are reported and the mean
IoU is a fraction.

### AStepOfTwoOutputsIsRefused

A step that published two outputs fails with `InvalidDescription` naming "published 2".

### ABatchWithNoMaskIsRefused

A batch with no mask role fails with `InvalidDescription` naming `'mask'`.

### LogitsOfAnotherCountAreRefused

Extents of four classes against logits of three fail with `Incompatible` naming "the logits".

### MoreThanTheMostClassesTheMatrixCountsIsRefused

Extents of 4097 classes fail with `InvalidDescription` naming "4097 classes" and "at most 4096".

### LogitsTheHostCannotReadAreRefused

Int32 logits fail with `InvalidDescription` naming "the logits" and "can be read back".

### ALabelOutOfBoundsIsRefused

A mask holding 3 at pixel 1 against three classes fails with `Incompatible` naming "true class of
pixel 1" and "is 3".

Every refusal leaves the metrics empty.
