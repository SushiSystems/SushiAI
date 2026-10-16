# tests/vision/detect/free/test_assigner_config.cpp


Suites: `Small_DetectAssignerConfig`.

Tests that `one_to_many_assigner_config` and `one_to_one_assigner_config` carry the class count
and refuse a count below one.

This file needs no device. `tests/vision/detect/free/test_assigner.cpp` compares both presets
and the member defaults of `AssignerConfig` with the reference settings, and checks how
`stride_val` follows the strides. This file adds the class count, which those cases fix at 80.

## K_STRIDES

`K_STRIDES` is 8, 16 and 32, the strides of a three-level model.

## BothPresetsCarryTheClassCount

Both presets built for 3 classes hold `num_classes` 3.

## BothPresetsRefuseAClassCountBelowOne

The one-to-many preset refuses 0 classes and the one-to-one preset refuses -2. Each message asks
for a positive class count and names the count it got.
