# tests/vision/training/test_pixel_losses.cpp


Suites: `Small_VisionTrainingPixelLosses`.

Tests the value and the operations `emit_pixel_cross_entropy` and `emit_soft_dice` add to a host
graph.

This file needs no device: each case builds a graph and reads it back. The objective tests,
`tests/vision/training/test_pixel_cross_entropy_objective.cpp` and
`tests/vision/training/test_dice_objective.cpp`, reach both emitters through a model and compare
the losses and gradients with a host computation. This file calls the emitters directly.

## count_nodes

`count_nodes` returns the number of nodes of a graph that run one operation.

## TheCrossEntropyIsOneValueOfTheLogitsType

Float64 logits of 6 rows and 3 classes give a 1-element float64 loss built from one K_LOG_SOFTMAX
and one K_GATHER_ROWS, with no K_NLL_LOSS and so no one-hot.

## TheSoftDiceIsOneValueThatGathersAndScattersOnce

Float32 logits and the five `DiceOperands` give a 1-element float32 loss with one K_GATHER_ROWS
and one K_SCATTER_ADD_ROWS.

## DefaultOperandsNameNoValueAndTheEpsilonIsAMillionth

Every field of a default `DiceOperands` is `K_NO_VALUE`, and `K_DICE_EPSILON` is 1e-6.
