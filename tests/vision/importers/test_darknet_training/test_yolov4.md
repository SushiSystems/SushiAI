# tests/vision/importers/test_darknet_training/test_yolov4.cpp


Suites: `Integration_DarknetYolov4Training`, in `sushiai_vision_importers_tests`.

The fixture `DarknetTrainingTest` reads `yolov4.cfg`, `yolov4.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov4_plain_pool`, one image of
YOLOv4 with its deltas and assignments, and `yolov4_train_step`, one training step of two images.
The box loss is `BoxLoss::CompleteIoU` and the objectness target is `ObjectnessTarget::One`. Every
case is skipped without its files.

The run of 2026-10-13 on this model, the first under the default memory limit, set the bounds of
the step cases by the owner's ruling of that day: each bound that run exceeded is now about 1.5
times the largest difference it measured. The heads were 0.205, 0.104 and 0.065 from the dump on
values near 40, the [yolo] outputs at most 0.0316, a box entry of the delta 0.0020 and any other
entry 0.0043, and an update of a head convolution 0.0047 of the dump's largest magnitude. The
cause is not known; YOLOv3 above differs by as much.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov4_plain_pool`. Its assignments
equal the dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of the dumped
delta and every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov4_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 1e-5 relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics, and the [yolo] output computed from it, are
within 1e-3 absolute plus 1e-4 relative of the dump.

## TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 1e-3. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1e-3 of the dump's largest magnitude plus 1e-3 of the
entry.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

## TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov4_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.
