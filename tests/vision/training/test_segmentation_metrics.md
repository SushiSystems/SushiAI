# tests/vision/training/test_segmentation_metrics.cpp


Suites: `Small_VisionTrainingSegmentationMetrics` and `Medium_VisionTrainingSegmentationMetrics`.

Tests `measure_segmentation` of `segmentation_metrics.hpp`: the mean IoU and the pixel accuracy a
per-pixel objective reports of a step, from the INT64 class value the step holds under
`pixel_classes`. The extents are one 2 by 2 map of three classes with the ignore index 255.

The `Small_` cases hand the function tensors from an `InMemoryTensorSource`. Such a tensor holds
its extents and element type and no element, and each refusal below is decided on those facts
before any element is read. The `Medium_` cases allocate tensors on the CPU SYCL device through
`EngineTest`, because they read the mask and the class values.

### TheMetricsEqualTheHandCount

The step holds the INT64 class values 0, 1, 0, 1 under `pixel_classes` against the mask 0, 2, 255,
1, so two of three counted pixels are right. Class 0 has an IoU of 1, class 1 of 1 / 2 and class
2 of 0, so the mean IoU is 0.5 and the pixel accuracy 2 / 3, each to 16 double unit roundoffs.
Two metrics are reported and the mean IoU is a fraction.

### MoreThan4096ClassesAreCounted

Extents of 4100 classes over four pixels: the class values 0, 4099, 0, 2 against the mask 0, 4099,
4099, 2. Three of four are right, so the pixel accuracy is 3 / 4. Classes 0 and 4099 each have an
IoU of 1 / 2 and class 2 of 1, so the mean IoU is 2 / 3. The matrix of this case holds 4100 by
4100 int64 counts, 134 MB, which is why the case is `Medium_`.

### AStepWithNoClassValueIsRefused

A step that holds no objective value fails with `InvalidDescription` naming `'pixel_classes'` and
"holds no objective value".

### ABatchWithNoMaskIsRefused

A batch with no mask role fails with `InvalidDescription` naming `'mask'`.

### AClassValueOfAnotherCountIsRefused

A class value of 3 elements against extents of 4 pixels fails with `Incompatible` naming "class
value holds 3 elements".

### AClassValueOfAnotherTypeIsRefused

An INT32 class value of 4 elements fails with `Incompatible` naming "class value holds 4 elements
of int32".

### ALabelOutOfBoundsIsRefused

A mask holding 3 at pixel 1 against three classes fails with `Incompatible` naming "true class of
pixel 1" and "is 3".

### AClassValueOutOfBoundsIsRefused

A class value of 3 at pixel 1, whose truth 2 is a class, against three classes fails with
`Incompatible` naming "predicted class of pixel 1" and "is 3".

Every refusal leaves the metrics empty.
