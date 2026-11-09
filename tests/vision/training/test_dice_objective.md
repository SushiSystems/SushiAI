# tests/vision/training/test_dice_objective.cpp


Suites: `Small_DiceObjective`, `Small_DiceObjectiveExtents` and `Medium_DiceObjectiveEndToEnd`.

Tests `DiceObjective` of `dice_objective.hpp`: the mean over the present classes of one minus the
soft Dice of the softmax probabilities. The graph cases read a model of one biased 1 by 1
convolution from 5 channels to 3 classes over a batch of 2 maps of 4 by 4, so a batch holds 32
pixels. They build the graph on the host and need no device.

### TheDiceDeclaresTwoIndexLeavesAndNoOneHot

`declare` adds five leaves: `score_indices` and `class_indices` as INT32 `[32]`, `kept` as
`[32, 1]`, and `class_pixels` and `class_weights` as `[3]`. No `one_hot` leaf exists, so no leaf
is `[R, C]`. The objective requires the `mask` role.

### TheDiceLossIsDifferentiated

The training graph's loss has one element, it has no target value, and both parameters, the weight
and the bias, have a gradient. The number of gather and scatter nodes is not asserted.

### MakeRefusesAnExtentBelowOne

Rows: a batch, a height, a width and a class count of 0. Each fails with `InvalidDescription`
naming "dice" and "at least 1".

### TheDiceLossAndItsGradientEqualTheHostComputation

On an engine, a 2 by 3 map of four classes with two ignored pixels and one absent class gives a
loss equal to the host Dice, and a gradient equal to the central difference of the host Dice, each
within 256 float32 unit roundoffs. The host computation is independent of the graph.
