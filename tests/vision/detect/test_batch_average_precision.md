# tests/vision/detect/test_batch_average_precision.cpp


Suites: `Small_BatchAveragePrecision`.

Tests `add_batch` against a mean computed by hand. This file needs no device.

## K_CLASS0_AP

`K_CLASS0_AP` is 0.495: with recall 0.5 and a closing point at 0 the envelope is 1 up to recall 0.5
and 0 above, so 49 trapezoids of 0.01 at 1 and one that falls from 1 to 0, 0.49 + 0.005.

## MeansAClassWithAMissAndAFalsePositiveWithAFoundOne

Two images, three boxes and a padding slot. Class 0 has two boxes: one is found at score 0.9, one
is missed, and a false positive at 0.8 sits in the second image. Class 1 has one box, found. AP at
IoU 0.5 is 0.495 for class 0, 0.995 for class 1 and 0.745 for the mean. The instance count of 3
shows the padding slot did not enter as a fourth box.

## BatchesAccumulateIntoOneMetric

Two batches of one image add up as two images do: one box found and one image with a box and no
detection give recall 0.5 and the AP 0.495 of class 0.

## RejectsAnArrayCountThatIsNotTheBatch

One detection array for a batch of two images returns `InvalidDescription` and leaves the metric
empty.
