# tests/vision/importers/test_darknet_training/test_yolov7.cpp


Suites: `Large_DarknetYolov7Training` and `Small_DarknetYolov7TrainingPlan`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov7.cfg`, `yolov7.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov7_plain_pool_thresh_0p01`, one
image of YOLOv7 with its deltas and assignments, and `yolov7_train_step`, one training step of two
images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is
`ObjectnessTarget::BestIoU`. Every case is skipped without its files.

The run of 2026-10-13 on this model, the first under the default memory limit, set three bounds
of the step cases by the owner's ruling of that day, each about 1.5 times the largest difference
that run measured: the heads were at most 0.0052 from the dump, the [yolo] outputs the same, and
an entry of the delta that is no box entry 0.00103. The other bounds held. The cause of the
differences is not known.

Every bound below is a multiple of the float unit roundoff u, 2^-24: the smallest power
of two not tighter than the decimal bound the measured differences set.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov7_plain_pool_thresh_0p01`. Its
assignments equal the dumped ones exactly, a box entry is within 32 u absolute plus 2048 u relative of
the dumped delta and every other entry within 32 u, and the stored cost agrees to 256 u relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov7_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 256 u relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics, and the [yolo] output computed from it, are
within 262144 u absolute plus 2048 u relative of the dump.

## TheDeltasOfTheStepMatchDarknet

The check is the one [`test_yolov4.md`](test_yolov4.md) describes, run on this model. A box entry is
within 32768 u absolute plus 4096 u relative, any other entry within 32768 u, and the costs to
32768 u relative.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 32768 u of the dump's largest magnitude plus 32768 u of the
entry.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 256 u of the dump after one blend. The variance after the blend is checked twice at the
same 256 u, each time against a variance the test computes on the host from the batch and the
The check is the one [`test_yolov3.md`](test_yolov3.md) describes, run on this model.

## TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov7_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.

## ThePlansOfTheStepOverlapLiveRanges

A compact YOLOv7 cfg is held in the test as text: swish convolutions, a four-way concatenation, a
stride-2 pool, an upsample and two heads, on a 64 by 64 input with one class. `check_training_plan`
translates it in the Separate form, reads the grid of each head from the model graph, traces the
training graph of a step of two images and builds its memory plan, all on the host. The test prints
the graph's nodes and the bytes the plan asks for, and expects that the plan overlaps live ranges.
It reads no file, so it is the only case of this file that runs without the Darknet reference data;
the cfg is not `yolov7.cfg`.
