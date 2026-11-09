# tests/vision/training/test_pixel_labels.cpp


Suites: `Small_VisionTrainingPixelLabels` and `Medium_VisionTrainingPixelLabels`.

Tests `validate_pixel_label_values` with an ignore index the caller chooses, and
`validate_pixel_labels` with a mask of rank three.

The ignore index case reads labels from a vector, so it needs no device and is small. The rank
three case allocates a tensor on the CPU SYCL device through `EngineTest`, so its suite is medium.
`tests/vision/training/test_pixel_targets.cpp` checks a rank-two mask with ignore index 255: a
label above the classes, a negative label, a wrong pixel count and a real mask. This file covers
another ignore index and another rank.

## Medium_VisionTrainingPixelLabels

The fixture builds INT32 tensors of class indices on the test's engine.

## TheCallersIgnoreIndexReplacesTheDefault

In `Small_VisionTrainingPixelLabels`. With ignore index -1, a mask holding -1 passes and a mask holding 255 is refused, naming the value
and pixel 1.

## AMaskOfRankThreeIsCheckedByItsElementCount

A 2 by 2 by 2 mask passes for 8 pixels and is refused for 4, with a message that names the 4 it
expects.
