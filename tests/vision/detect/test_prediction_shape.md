# tests/vision/detect/test_prediction_shape.cpp


Suites: `Small_DetectPredictionShape`.

Tests the default extents of a `PredictionShape` and the order of its fields.

This file needs no device. The header declares a struct of three extents and no function, so
the cases pin its defaults and the order a braced initialiser fills.

## ADefaultShapeHasNoExtents

A default `PredictionShape` has a batch, an anchor count and a class count of 0.

## ABracedShapeNamesImagesThenAnchorsThenClasses

`PredictionShape{2, 8400, 80}` holds 2 images, 8400 anchors and 80 classes, in that order.
