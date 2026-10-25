# tests/vision/detect/free/test_free_task.cpp

Suites: `Small_DetectFreeTask`.

Tests that `read_free_task` and `write_free_task` convert an anchor-free task both ways
without loss and refuse what the section does not allow. This file needs no device.

## ATaskRoundTripsThroughItsSection

A task whose floats no short decimal states is written, printed as JSON, read back and equals
the original bit for bit.

## ASectionReadsFieldByFieldAndWritesBack

The full section `K_FULL_SECTION` reads into the expected fields and writes back to the same
JSON text.

## AnAbsentAssignmentKeepsTheBranchPresets

A section without `assignment` keeps both branch presets; one-to-one is top 7 then 1.

## RefusesTheTypeAndKeysItDoesNotKnow

Another type, an unknown key (`strides`), a missing `threshold` and a branch without `top_k2`
are refused, each naming the key.

## RefusesValuesOutOfRange

A score threshold of 1.5, an exponent under `iou`, a `top_k2` above `top_k` and a negative
`beta` are refused, each naming the key.
