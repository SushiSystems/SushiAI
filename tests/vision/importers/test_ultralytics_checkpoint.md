# tests/vision/importers/test_ultralytics_checkpoint.cpp

Suites: `Medium_UltralyticsCheckpoint`, `Small_UltralyticsCheckpoint`.

The `Medium_` cases read `model.pt` of `tests/vision/registration/fixtures/yolo26/`, which
`tools/reference/export_yolo26_reference.py` writes, and skip when it has not run.

## ItReadsTheDescriptionOfScaleN

The description holds the scale "n", which the file name `yolo26n.yaml` selects.

## ItNamesTensorsByTheirPathBelowTheLayerList

The first tensor is `0.conv.weight`, the weight of the first Conv below the layer list.

## ItLeavesOutEveryStepCounter

The 114 `num_batches_tracked` counters, one per normalisation, are listed as left out.

## BytesThatAreNoArchiveAreRefused

Sixty-four bytes that are no zip archive fail with `InvalidFile`, naming the file.
