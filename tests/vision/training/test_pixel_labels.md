# tests/vision/training/test_pixel_labels.cpp


Suites: `Medium_VisionTrainingPixelLabels`.

Tests `validate_pixel_labels` with an ignore index the caller chooses and with a mask of rank
three.

The cases allocate tensors on the CPU SYCL device through `EngineTest`, so the suite is medium.
`tests/vision/training/test_pixel_targets.cpp` checks a rank-two mask with ignore index 255: a
label above the classes, a negative label, a wrong pixel count and a real mask. This file covers
another ignore index and another rank.

## Medium_VisionTrainingPixelLabels

The fixture builds INT32 tensors of class indices on the test's engine.

## TheCallersIgnoreIndexReplacesTheDefault

With ignore index -1, a mask holding -1 passes and a mask holding 255 is refused, naming the value
and pixel 1.

## AMaskOfRankThreeIsCheckedByItsElementCount

A 2 by 2 by 2 mask passes for 8 pixels and is refused for 4, with a message that names the 4 it
expects.
