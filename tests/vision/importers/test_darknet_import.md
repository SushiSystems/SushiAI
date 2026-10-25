# tests/vision/importers/test_darknet_import.cpp


Suites: `Medium_DarknetImport`.

Tests `import_darknet_model` and `import_darknet_files` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_import.hpp`. The sample
`.cfg` text describes a 32 by 32 network of three channels: a normalised convolution of 4
filters, a plain convolution of 18 filters and one `[yolo]` head of one class with the mask
0, 1, 2 over the anchors 4,8, 8,16 and 16,24. The cases parse it with `parse_darknet_config`
and translate it with `translate_darknet_model`.

## KeepsTheRowsAndTheOutputsOfTheTranslation

Imported without weights, the model has the translation's input shape and the same type in
every row. It has one output, named "head.0", at the node the translation's first output
names.

## StatesTheHeadsAsAnAnchorDetectionTask

The task section has the type "anchor_detection" and reads through `read_anchor_task` as one
class and one head on the output "head.0" with `HeadCoding::Logit` and the mask {0, 1, 2}.
It has three anchors; the first has the width 4/32 and the third the height 24/32, within
1e-6.

## HoldsNoTensorWithoutWeights

An import with a null weights pointer has an empty `weights` list.

## NamesEveryTensorAsItsTargetDoes

With weights of the lengths the translation's shapes state, the names of the imported tensors
are the set of the non-empty names in the translation's targets, and there are seven tensors.

## CarriesTheRecipeAndItsNotes

The imported `training` value has the keys `objective`, `optimizer`, `schedule`, and the
import carries five notes.

## AnotherCountOfFilesIsRefused

`import_darknet_files` fails with `InvalidDescription` for no file ("'darknet'", "0 files")
and for three files (".cfg", "3 files").

## AMissingCfgIsRefusedByName

`import_darknet_files` of a `.cfg` path that does not exist fails with `FileNotFound` and the
path's name.

Suites: `Medium_DarknetImportEndToEnd`, in `sushiai_vision_importers_tests`.

The suite runs once for each model of the parity table in `modules/vision/darknet_forward_cases/source/darknet_forward.cpp`:
`yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7`. The fixture
`DarknetImportTest` imports the model in memory from its `.cfg` and `.weights` in the directory
`SA_REFERENCE_DATA_DIR` names. A case is skipped when a file of its model is missing.

## TheImportedRowsAndTaskAreThoseOfTheTranslation

The imported model file, without its task, is the text of the translated model. The task reads back
as the anchors, the heads and the suppression criterion.

## TheImportedWeightsAreThoseOfTheDirectStore

A state filled from the imported tensors equals, byte for byte, the state `store_darknet_weights`
fills. The imported tensors supply every entry exactly once.

## TheWrittenFilesLoadAsTheImport

`check_written_import` writes the import into a scratch directory: three files. The written
`model.json`, read back and written again, is the text of the imported model; the written
`train.json` is the text `compose_training_file` gives. `load_checkpoint` fills a state built
from the read-back model with the bytes of the state `store_darknet_weights` fills.
