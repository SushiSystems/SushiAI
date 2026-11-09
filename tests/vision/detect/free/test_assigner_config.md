# tests/vision/detect/free/test_assigner_config.cpp


Suites: `Small_DetectAssignerConfig` and `Medium_DetectAssignerConfig`.

Tests that `one_to_many_assigner_config` and `one_to_one_assigner_config` carry the class count,
take a single stride as `stride_val` and refuse a count below one.

This file needs no device. `tests/vision/detect/free/test_assigner.cpp` compares both presets
with the reference settings, and checks how `stride_val` follows the strides of the one-to-many
preset. This file adds the class count, which those cases fix at 80, and the single stride of the
one-to-one preset.

## K_STRIDES

`K_STRIDES` is 8, 16 and 32, the strides of a three-level model.

## BothPresetsCarryTheClassCount

Both presets built for 3 classes hold `num_classes` 3.

## BothPresetsTakeASingleStrideAsStrideVal

Both presets built over the one stride 4 hold `stride_val` 4.

## BothPresetsRefuseAClassCountBelowOne

In `Medium_DetectAssignerConfig`: the refusal is a stop, which `SA_EXPECT_FATAL` sees only in a
child process, and a death test is medium.

The one-to-many preset refuses 0 classes and the one-to-one preset refuses -2. Each message asks
for a positive class count and names the count it got.
