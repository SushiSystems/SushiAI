# tests/vision/training/test_pixel_cross_entropy_objective.cpp


Suites: `Small_PixelCrossEntropyObjective`, `Small_PixelCrossEntropyObjectiveExtents`,
`Small_PixelCrossEntropyObjectiveModels`, `Small_PixelCrossEntropyObjectiveFeed`,
`Medium_PixelCrossEntropyObjectiveEndToEnd` and `Medium_SegmentOverfit`.

Tests `PixelCrossEntropyObjective` and the two factories of `pixel_cross_entropy_objective.hpp`
and `dice_objective.hpp` that read a pixel objective from a model. The `Small_` cases build a
graph or a context on the host: a model of one biased 1 by 1 convolution from 5 channels to 3
classes over 2 maps of 4 by 4, so a batch holds 32 pixels. The `Medium_` cases train on the CPU
SYCL device through `EngineTest`.

### MakeKeepsWhatItIsGiven

`make` keeps the extents (32 pixels, 3 classes), the ignore index 7, and the name
`pixel_cross_entropy`.

### MakeRefusesAnExtentBelowOne

Rows: a batch, a height, a width and a class count of 0. Each fails with `InvalidDescription`
naming "pixel_cross_entropy" and "at least 1". The Dice objective's own file keeps its rows.

### MakeRefusesAnIgnoreIndexThatIsAClass, MakeRefusesAnIgnoreIndexThatIsAClassForTheDice

An ignore index of 2 against three classes fails for the cross entropy, naming "ignore index 2" and
"3 classes". An ignore index of 0 fails for the Dice objective, naming "ignore index 0".

### MakeAcceptsAnIgnoreIndexOutsideTheClasses

The ignore indices 3 and -1 are accepted.

### MakeRefusesAWeightTypeThatIsNoReal

A weight type of int32 fails with `InvalidDescription` naming "float32".

### TheCrossEntropyDeclaresIndicesAndWeights

`declare` adds `targets` as INT32 `[32]` and `weights` as float32 `[32, 1]`, and requires the INT32
`[4, 4]` mask role with no class count.

### TheCrossEntropyIsOneElementWithGradients

The training graph's loss has one element, its targets are INT32, and both parameters have a
gradient. The operations that make the loss are not counted.

### TheCrossEntropyFactoryReadsTheModelsOutput, TheDiceFactoryReadsTheModelsOutputAndItsIgnoreIndex

The cross entropy factory builds an objective named `pixel_cross_entropy` of 32 pixels and the
default ignore index 255. The Dice factory reads the ignore index 9 from its block.

### TheCrossEntropyFactoryRefusesAnUnknownKey, TheDiceFactoryRefusesARealIgnoreIndex, TheCrossEntropyFactoryRefusesAClassAsIgnoreIndex

A block with the key `weight` fails naming "the objective block", `'weight'` and `'ignore_index'`.
An ignore index of 0.5 fails for the Dice factory naming `'ignore_index'` and "whole number". An
ignore index of 1 fails for the cross entropy factory naming "ignore index 1".

### TheFactoryRefusesAModelThatDoesNotFit

Rows for the cross entropy factory: a model that publishes two outputs ("one [N, H, W, classes]
output", "publishes 2"), an output of `[2, 3]`, and an output half the size of the input. Each
fails with `InvalidDescription`.

### TheDiceFactoryRefusesMoreClassesThanTheModelScores

Data of 4 classes against a model that scores 3 fails naming "holds 4 classes" and "scores 3".

### TheCrossEntropyAndItsGradientEqualTheHostComputation

On an engine, a 2 by 3 map of four classes with two ignored pixels gives a loss equal to the host
cross entropy and a gradient equal to `(softmax - one_hot) / kept` on a kept pixel and zero on an
ignored one, each to 256 float32 unit roundoffs.

### AWeightedSumOfBothLossesEqualsTheHostComputation

A weighted sum of the two objectives gives a loss and a gradient equal to the host computation.

### AMaskWithNoKeptPixelGivesALossOfZero, AMaskWithNoKeptPixelGivesNoGradient

A mask of ignored pixels alone gives a loss of 0 and a gradient of zeros.

### ALabelOutsideTheClassesStopsTheFeed

A label at the class count stops the feed and names its pixel. The case builds its state and pool on
the engine of a `HostExecutionBackend` and the feed refuses before any task runs, so it is `Small_`.

### TheLossFallsOnOneBatch, TheMeanIouRisesOnOneBatch

A tiny segmenter of two convolutions and a head, trained on one batch, ends with a loss below a
fraction of its first and a mean IoU above the bound. The step count, the compile count and the
relation between the accuracy and the mean IoU are not asserted here; the metric belongs to
`test_segmentation_metrics.cpp`. The fraction and the bound are marked unconfirmed in the test.
