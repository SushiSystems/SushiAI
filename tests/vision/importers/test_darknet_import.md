# tests/vision/importers/test_darknet_import.cpp


Suites: `Small_DarknetImport`, `Small_DarknetImportFileCount` and `Medium_DarknetImport`.

Tests `import_darknet_model` and `import_darknet_files` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_import.hpp`. The sample
`.cfg` text describes a 32 by 32 network of three channels: a normalised convolution of 4
filters, a plain convolution of 18 filters and one `[yolo]` head of one class with the mask
0, 1, 2 over the anchors 4,8, 8,16 and 16,24. The cases parse it with `parse_darknet_config`
and translate it with `translate_darknet_model`. The `Small_` cases import in memory: they
open no file, start no thread and use no device.

## KeepsTheRowsOfTheTranslation

Imported without weights, the model has the translation's input shape and the same list of row
types, compared as two vectors.

## KeepsTheOutputsOfTheTranslation

The import has one output, named "head.0", at the node the translation's first output names.

## StatesTheHeadsAsAnAnchorDetectionTask

The task section has the type "anchor_detection" and reads through `read_anchor_task` as one
class and one head on the output "head.0" with `HeadCoding::Logit` and the mask {0, 1, 2}.
It has three anchors; the first has the width 0.125 (4/32) and the third the height 0.75
(24/32), both exact in float and compared for equality.

## HoldsNoTensorWithoutWeights

An import with a null weights pointer has an empty `weights` list.

## NamesEveryTensorAsItsTargetDoes

With weights of the lengths the translation's shapes state, the set of the imported tensor
names equals the set of the non-empty names in the translation's targets, and there are seven
tensors.

## CarriesTheRecipe

The imported `training` value has the keys `objective`, `optimizer`, `schedule`.

## CarriesTheNotes

The import carries five notes.

## Small_DarknetImportFileCount

`import_darknet_files` fails with `InvalidDescription` before it opens a file. One row each:
`NoFile` for no path ("'darknet'", "0 files") and `ThreeFiles` for three paths (".cfg",
"3 files").

## AMissingCfgIsRefusedByName

`Medium_`. `import_darknet_files` of a `.cfg` path that does not exist fails with
`FileNotFound` and the path's name.

Suites: `Medium_DarknetImportEndToEnd`, in `sushiai_vision_tests`.

The suite runs once for each model of the parity table in `modules/vision/darknet_forward_cases/source/darknet_forward.cpp`:
`yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7`. The fixture
`DarknetImportTest` imports the model in memory from its `.cfg` and `.weights` in the directory
`SA_REFERENCE_DATA_DIR` names. A case is skipped when a file of its model is missing.

## TheImportedRowsAreThoseOfTheTranslation

`check_imported_rows`: the imported model file, without its task, is the text of the translated
model.

## TheImportedTaskIsThatOfTheTranslation

`check_imported_task`: the task reads back as the anchors, the heads and the suppression
criterion.

## TheImportedWeightsAreThoseOfTheDirectStore

`check_imported_weights`: a state filled from the imported tensors equals, byte for byte, the
state `store_darknet_weights` fills.

## TheImportedTensorsSupplyEveryEntryOnce

`check_imported_entries`: `check_tensors_against_entries` accepts the imported tensors against
the entries of a state built from the imported model.

## TheWrittenFilesReadBackAsTheImport

`check_written_files` writes the import into a scratch directory: three files. The written
`model.json`, read back and written again, is the text of the imported model; the written
`train.json` is the text `compose_training_file` gives.

## TheWrittenWeightsLoadAsTheDirectStore

`check_written_weights` writes the import into its own scratch directory. `load_checkpoint` fills
a state built from the read-back model with the bytes of the state `store_darknet_weights` fills.
