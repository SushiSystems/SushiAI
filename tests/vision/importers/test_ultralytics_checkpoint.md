# tests/vision/importers/test_ultralytics_checkpoint.cpp

Suites: `Medium_UltralyticsCheckpoint`, `Medium_UltralyticsWeightFile`, `Small_UltralyticsCheckpoint`.

The `Medium_` cases read `model.pt` of `tests/vision/registration/fixtures/yolo26/`, which
`tools/reference/export_yolo26_reference.py` writes, and skip when it has not run. The `Medium_UltralyticsWeightFile` case reads `weights.sushi` of the
same folder and skips when it is absent, as it is under `--external-weights`.

## ItReadsTheDescriptionOfScaleN

The description holds the scale "n", which the file name `yolo26n.yaml` selects.

## ItNamesTensorsByTheirPathBelowTheLayerList

The first tensor is `0.conv.weight`, the weight of the first Conv below the layer list.

## ItLeavesOutEveryStepCounter

The 114 `num_batches_tracked` counters, one per normalisation, are listed as left out.

## TheExportedWeightFileIsVersionTwoWithNoRunState

`weights.sushi` opens with `IO::CheckpointReader` as format version `K_CHECKPOINT_VERSION` (2),
with a run-state count of 0 and at least one model-state record. A file the exporter wrote before
it wrote version 2 states version 1 and fails the case.

## BytesThatAreNoArchiveAreRefused

Sixty-four bytes that are no zip archive fail with `InvalidFile`, naming the file.
