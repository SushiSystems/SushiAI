# tests/vision/importers/test_ultralytics_import/test_import.cpp

Suites: `Medium_UltralyticsImport`, `Small_UltralyticsImport`.

The `Medium_` cases read `model.pt` and `weights.sushi` of
`tests/vision/registration/fixtures/yolo26/`, which `tools/reference/export_yolo26_reference.py`
writes from one seeded YOLO26n of the pinned Ultralytics commit: `model.pt` is the model as
`torch.save` writes a checkpoint, `weights.sushi` its tensors as the exporter lays them out. A
case skips when the exporter has not run.

## ItImportsEveryFloatingTensorOfTheCheckpoint

The import holds 594 tensors, the floating state-dict entries `MANIFEST.json` lists.

## EachTensorIsTheExportedOneBitForBit

Every imported tensor equals, in shape and byte for byte, the record of `weights.sushi` named
"model." and its name.

## TheModelIsTheHandWrittenYolo26ModelFile

The imported model writes the same text as `applications/examples/yolo26/model.json`.

## TwoFilesAreRefused

Two files fail with `InvalidDescription`, naming the format and the count, before either is opened.
