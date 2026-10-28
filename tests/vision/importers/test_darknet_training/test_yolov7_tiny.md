# tests/vision/importers/test_darknet_training/test_yolov7_tiny.cpp


Suites: `Medium_DarknetYolov7TinyTraining`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov7-tiny.cfg`, `yolov7-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov7_tiny_plain_pool_thresh_0p01`,
one image of YOLOv7-tiny with its deltas and assignments, and `yolov7_tiny_train_step`, one training
step of two images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is
`ObjectnessTarget::BestIoU`. Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. `test_darknet_yolov3_tiny_training.cpp`
above gives the two differences from the reference that cause them.

## TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

## TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov7_tiny_plain_pool_thresh_0p01`. Its
assignments equal the dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of
the dumped delta and every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

## TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov7_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

## TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 2e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 3e-2 absolute plus 1e-4
relative. The largest differences measured were 0.0128 and 0.0197.

## TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 7.5e-3; the largest differences measured were 0.00094
and 0.0049. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

## TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 7e-3 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 4.4e-3 of that magnitude.

## TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

## TheRollingStatisticsAfterOneStepMatchDarknet

The check is the one [`test_yolov3.md`](test_yolov3.md) describes, run on this model.

## TheCostOfOneBatchFallsUnderAThird

200 steps on the batch and the truth of `yolov7_tiny_train_step` bring the stored cost of the
Separate model below 0.33 of the first step's.

The share is 0.33 where the other five models state 0.1, by the owner's ruling of 2026-10-12.
CPU Darknet, run for 200 steps on the same batch, weights and learning rate with no burn-in,
takes its summed layer cost from 30.72 to 6.93, which is 0.226 of the first, and its lowest
step holds 4.27. SushiAI goes from 30.71 to 6.63, which is 0.216. Neither reaches a tenth.
The two curves agree at step 0 and are not compared step by step after it. Why the cost
stops there is not known.
