# tests/vision/detect/test_suppression_section.cpp

Suites: `Small_DetectSuppressionSection`.

Tests that `read_suppression_section` and `write_suppression_section` convert the `suppression`
object of a task section and refuse what it does not allow, naming the key. This file needs no
device.

## ADistanceIoUObjectReadsItsThreeFields

An object of `distance_iou`, 0.625 and 0.5 reads into those three fields.

## AnIoUObjectIsWrittenWithoutTheExponent

An `iou` section writes JSON that holds no "distance_exponent".

## RefusesAnAbsentObject

A task without "suppression" fails, naming it.

## RefusesAnUnknownCriterion

The criterion "giou" fails, naming "'criterion'", "'giou'" and the choices.

## RefusesAThresholdAboveOne

A threshold of 1.5 fails, naming "'threshold'" and "1.5".

## RefusesAnExponentOnIoU

A "distance_exponent" under `iou` fails, naming it and "'iou'".

## RefusesAMissingExponentOnDistanceIoU

`distance_iou` without "distance_exponent" fails, naming it.
