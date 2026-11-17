# tests/vision/training/test_pixel_losses.cpp


Suites: `Small_VisionTrainingPixelLosses`.

Tests the value `emit_pixel_cross_entropy` and `emit_soft_dice` add to a host graph.

This file needs no device: each case builds a graph and reads it back. The objective tests,
`tests/vision/training/test_pixel_cross_entropy_objective.cpp` and
`tests/vision/training/test_dice_objective.cpp`, reach both emitters through a model and compare
the losses and gradients with a host computation. This file calls the emitters directly. The
values of the losses are those of the objective tests; the two last cases count only the nodes that
read a label.

`DiceOperands` is plain data and `K_DICE_EPSILON` is a constant, so neither has a case.

## TheCrossEntropyIsOneValueOfTheLogitsType

Float64 logits of 6 rows and 3 classes give a 1-element float64 loss.

## TheSoftDiceIsOneFloat32Value

Float32 logits and the four `DiceOperands` give a 1-element float32 loss.

## TheCrossEntropyPicksEachLabelWithOneTakeAlong

The graph of `emit_pixel_cross_entropy` holds one `K_TAKE_ALONG` node and no `K_GATHER_ROWS` node.

## TheSoftDicePicksEachLabelWithOneTakeAlong

The graph of `emit_soft_dice` holds one `K_TAKE_ALONG` node, one `K_SCATTER_ADD_ROWS` node and no
`K_GATHER_ROWS` node.
