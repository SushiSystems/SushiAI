# tests/vision/detect/free/test_free_task.cpp

Suites: `Small_DetectFreeTask`.

Tests that `read_free_task` and `write_free_task` convert an anchor-free task both ways
without loss and refuse what the section does not allow. This file needs no device.

## ATaskRoundTripsThroughItsSection

A task whose floats no short decimal states is written, printed as JSON, read back and equals
the original bit for bit.

## ASectionReadsFieldByField

The full section `K_FULL_SECTION` reads into the expected fields.

## ASectionWritesBackToTheSameJson

The task read from `K_FULL_SECTION` writes back to the same JSON text.

## AnAbsentAssignmentKeepsTheBranchPresets

A section without `assignment` keeps both branch presets; one-to-one is top 7 then 1.

## AnIoUSectionKeepsTheDefaultDistanceExponent

The same section, under `iou`, reads the default distance exponent.

## RefusesAnotherType

A type other than `anchor_free_detection` is refused, naming `type` and the type it wants.

## RefusesAnUnknownKey

An unknown key (`strides`) is refused, naming it.

## RefusesAMissingThreshold

A suppression without `threshold` is refused, naming it.

## RefusesAMissingTopK2

A branch without `top_k2` is refused, naming it.

## RefusesAScoreThresholdOutOfRange

A score threshold of 1.5 is refused, naming the key and the value.

## RefusesADistanceExponentOnIoU

A distance exponent under `iou` is refused, naming `distance_exponent` and `iou`.

## RefusesATopK2AboveTopK

A `top_k2` of 8 above a `top_k` of 7 is refused, naming the key and the value.

## RefusesANegativeBetaOnWrite

Writing a task whose one-to-one `beta` is negative is refused, naming `beta`.
