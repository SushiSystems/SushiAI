# tests/vision/importers/test_darknet_import/test_yolov4_forward.cpp


Suites: `Large_DarknetYolov4Forward`, in `sushiai_vision_importers_tests`.

The fixture `DarknetParityTest` imports YOLOv4 from `yolov4.cfg` and `yolov4.weights` in the
directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 608 pixel input with
the reference tensors under `references/`: `yolov4_plain_pool` at a threshold of 0.25 and
`yolov4_plain_pool_thresh_0p01` at 0.01. The model has three heads on grids of 76, 38 and 19, each
with three anchor slots over 80 classes, coded as logits. A head value is held to 1e-3 absolute plus
1e-4 relative, and a box entry or a score of a detection row to 1e-4. Every case is skipped when a
file is missing.

## TheFoldedHeadsMatchTheDumpedHeads

The three published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

## TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the logistic of every entry
but the sizes, with x and y stretched by the `scale_x_y` of 1.2, 1.1 and 1.05, and compared with the
dumped layer output within the head tolerance.

## TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: one at
0.25 and six at 0.01. Suppression runs at 0.45 on the IoU less the distance term to the power 0.6.
Both sides are sorted by class, then by score, and a row matches when its class is equal and its box
entries and score are within 1e-4. Every decision within 1e-6 of a threshold is printed.

## TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.
