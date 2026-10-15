# tests/vision/importers/test_darknet_training/test_yolov4_tiny.cpp


Suites: `Integration_DarknetYolov4TinyTraining`, in `sushiai_vision_importers_tests`.

The fixture `DarknetTrainingTest` reads `yolov4-tiny.cfg`, `yolov4-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov4_tiny`, one image of
YOLOv4-tiny with its deltas and assignments, and `yolov4_tiny_train_step`, one training step of two
images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is `ObjectnessTarget::One`.
Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. `test_darknet_yolov3_tiny_training.cpp`
above gives the two differences from the reference that cause them.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov4_tiny`. Its assignments equal the
dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of the dumped delta and
every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov4_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 7e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 8e-3 absolute plus 1e-4
relative. The largest differences measured were 0.0481 and 0.0051.

## TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 2e-3; the largest differences measured were 0.00043
and 0.0014. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1.3e-2 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 8.6e-3 of that magnitude.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The check is the one [`test_yolov3.md`](test_yolov3.md) describes, run on this model.

## TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov4_tiny_train_step` bring the stored cost of the
Separate model below 0.1 of the first step's.
