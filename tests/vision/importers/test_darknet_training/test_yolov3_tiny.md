# tests/vision/importers/test_darknet_training/test_yolov3_tiny.cpp


Suites: `Medium_DarknetYolov3TinyTraining`, in `sushiai_vision_importers_tests`.

The fixture `DarknetTrainingTest` reads `yolov3-tiny.cfg`, `yolov3-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov3_tiny`, one image of
YOLOv3-tiny with its deltas and assignments, and `yolov3_tiny_train_step`, one training step of two
images. The box loss is `BoxLoss::SquaredError` and the objectness target is
`ObjectnessTarget::One`. Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. Two differences from the reference
cause them, and neither is a difference of float32 rounding.

The first is the divisor of the batch variance. The library normalises a training batch by the
variance divided by the count and keeps the variance divided by the count less one for the rolling
statistic. The reference normalises by the second. On a 13 by 13 map of two images the count is 338,
so each normalised layer differs in scale by about 0.15 percent, and the heads end 0.0466 and 0.0334
from the dump. A host model of this network in double, written outside the repository and checked
against finite differences, sits 0.0018 and 0.0026 from the dump with the reference's divisor and
0.04664 and 0.03336 with the library's, against 0.04665 and 0.03337 measured here. The deltas and
the updates of the head convolutions inherit the difference.

The second is the reference's backward pass. Its convolution clears the delta of the layer before
it and then writes its own, so a delta that a route had already added to that layer is lost. In this
model the second head's delta through layers 18 and 17 never reaches layer 13. The dumped updates of
the first convolution are therefore not the gradient: the host model is 14.18 from the dumped weight
update with every path and 0.36 with that path removed, against a largest entry of 27.9. The library
keeps every path, and is 15.60 from the dump where the host model with the library's divisor is
15.61. The suite compares no update of the first convolution with the dump for that reason.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov3_tiny`. Its assignments equal the
dumped ones exactly, every entry is within 1e-6 of the dumped delta, and the stored cost agrees to
1e-5 relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov3_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 7e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 6e-3 absolute plus 1e-4
relative. The largest differences measured were 0.0466 and 0.0037.

## TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 2.5e-3 absolute plus
1.5e-4 relative, any other entry within 2.5e-3; the largest differences measured were 0.0017
and 0.0016. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 7e-3 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 4.2e-3 of that magnitude.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; the start of this section says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The check is the one [`test_yolov3.md`](test_yolov3.md) describes, run on this model.

## TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov3_tiny_train_step` bring the stored cost of the
Separate model below 0.1 of the first step's.

The stored cost is `HeadCost::stored` summed over the heads: the squared delta entries for a
squared-error head, and for a CIoU head the box weight times the mean of 1 - IoU plus the weighted
squares of the other entries. The summed squares of the delta are not used here, because the box
entries of a CIoU delta do not shrink as the boxes fit: on YOLOv4-tiny their squares stay between
1.0 and 2.8 over 300 steps while the stored cost falls from 15.9 to 0.03.
