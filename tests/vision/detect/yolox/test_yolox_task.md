# tests/vision/detect/yolox/test_yolox_task.cpp

Suites: `Small_DetectYoloxTask`.

Tests that `read_yolox_task` and `write_yolox_task` convert a YOLOX task both ways without loss
and refuse what the section does not allow, naming the key. This file needs no device.

## ATaskRoundTripsThroughItsSection

A task whose floats no short decimal states is written, printed as JSON, read back and equals
the original bit for bit.

## ASectionReadsFieldByField

The full section `K_FULL_SECTION` reads into the expected fields.

## ASectionWritesBackToTheSameJson

The task read from `K_FULL_SECTION` writes back to the same JSON text.

## AnIoUSectionKeepsTheDefaultDistanceExponent

A section under `iou` reads the default distance exponent.

## RefusesAnotherType

The type "anchor_free_detection" fails, naming "'type'" and "'yolox_detection'".

## RefusesAnUnknownKey

The key "strides" fails and is named.

## RefusesAnUnknownSuppressionKey

The key "class_agnostic" under "suppression" fails, naming it and "'suppression'".

## RefusesAMissingScoreThreshold

A section without "score_threshold" fails, naming it.

## RefusesAMissingThreshold

A suppression without "threshold" fails, naming it.

## RefusesAScoreThresholdOutOfRange

A score threshold of 1.5 fails, naming "'score_threshold'" and "1.5".

## RefusesANegativeScoreThresholdOnWrite

Writing a task whose score threshold is -0.5 fails, naming "'score_threshold'".
