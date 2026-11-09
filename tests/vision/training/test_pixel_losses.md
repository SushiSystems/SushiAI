# tests/vision/training/test_pixel_losses.cpp


Suites: `Small_VisionTrainingPixelLosses`.

Tests the value `emit_pixel_cross_entropy` and `emit_soft_dice` add to a host graph.

This file needs no device: each case builds a graph and reads it back. The objective tests,
`tests/vision/training/test_pixel_cross_entropy_objective.cpp` and
`tests/vision/training/test_dice_objective.cpp`, reach both emitters through a model and compare
the losses and gradients with a host computation. This file calls the emitters directly. The
operations that make a loss are not counted: the objective tests show that the loss is right.

`DiceOperands` is plain data and `K_DICE_EPSILON` is a constant, so neither has a case.

## TheCrossEntropyIsOneValueOfTheLogitsType

Float64 logits of 6 rows and 3 classes give a 1-element float64 loss.

## TheSoftDiceIsOneFloat32Value

Float32 logits and the five `DiceOperands` give a 1-element float32 loss.
