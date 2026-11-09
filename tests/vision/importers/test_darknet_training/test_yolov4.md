# tests/vision/importers/test_darknet_training/test_yolov4.cpp


Suites: `Large_DarknetYolov4Training` and `Small_DarknetYolov4TrainingPlan`, in `sushiai_vision_tests`.

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

Every bound below is a multiple of the float unit roundoff u, 2^-24: the smallest power
of two not tighter than the decimal bound the measured differences set.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov4_plain_pool`. Its assignments
equal the dumped ones exactly, a box entry is within 32 u absolute plus 2048 u relative of the dumped
delta and every other entry within 32 u, and the stored cost agrees to 256 u relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov4_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 256 u relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 8388608 u absolute plus 2048 u
relative of the dump, and the [yolo] output computed from it within 1048576 u absolute plus 2048 u
relative.

## TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 65536 u absolute plus
4096 u relative, any other entry within 131072 u. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 32768 u relative. The bounds do not cover a slot
that crosses the ignore threshold.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 131072 u of the dump's largest magnitude plus 32768 u of the
entry.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The check is the one [`test_yolov3.md`](test_yolov3.md) describes, run on this model.

## TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov4_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.

## ThePlansOfTheStepOverlapLiveRanges

A compact YOLOv4 cfg is held in the test as text: a mish backbone with a shortcut, the pooled
concatenation of the spatial pyramid, an upsample and two heads, on a 64 by 64 input with one class.
`check_training_plan` translates it in the Separate form, reads the grid of each head from the model
graph, traces the training graph of a step of two images and builds its memory plan, all on the
host. The test prints the graph's nodes and the bytes the plan asks for, and expects that the plan
overlaps live ranges. It reads no file, so it is the only case of this file that runs without the
Darknet reference data; the cfg is not `yolov4.cfg`.
