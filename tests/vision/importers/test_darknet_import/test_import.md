# tests/vision/importers/test_darknet_import/test_import.cpp

Suites: `Small_DarknetImport`, `Small_DarknetImportFileCount`, `Medium_DarknetImport`,
`Medium_DarknetImportEndToEnd`, in `sushiai_vision_tests`.

Tests the import of a Darknet model: its rows, task, weights and training files. The small suites
read a 32 by 32 network held in the test: one normalised convolution, one plain convolution and
one head. The end-to-end suite runs once per parity model through `check_imported_rows` and its
siblings of `darknet_import_cases`, and skips a case when a file of its model is missing.

## KeepsTheRowsOfTheTranslation

The imported rows are the rows of the translation, in order.

## KeepsTheOutputsOfTheTranslation

The imported outputs are the outputs of the translation.

## StatesTheHeadsAsAnAnchorDetectionTask

The task section reads back as the classes, anchors and head of the .cfg.

## HoldsNoTensorWithoutWeights

An import that is given no weight file holds no tensor.

## NamesEveryTensorAsItsTargetDoes

The tensors carry exactly the names the targets of the translation state.

## CarriesTheRecipe

The recipe is the one the .cfg's training settings state.

## CarriesTheAugmentationAsTheDataBlock

The augmentation keys of the .cfg arrive as the data keys of the darknet pipeline.

## CarriesTheNotes

The notes of the .cfg's training settings arrive with the import.

## IsRefused

A call of `import_darknet_files` with no file and one with three are refused, and the message
states the count.

## AMissingCfgIsRefusedByName

A .cfg that is not on disk is refused, and the message names its path.

## TheImportedRowsAreThoseOfTheTranslation

For each parity model, the model file without its task is the text of the translation.

## TheImportedTaskIsThatOfTheTranslation

The task reads back as the anchors, the heads and the criterion of the translation.

## TheImportedWeightsAreThoseOfTheDirectStore

A state filled from the imported tensors equals, byte for byte, the state `store_darknet_weights`
fills from the translation.

## TheImportedTensorsSupplyEveryEntryOnce

The imported tensors supply every entry of the state exactly once, and no other.

## TheWrittenFilesReadBackAsTheImport

The files the import writes read back as the import.

## TheWrittenWeightsLoadAsTheDirectStore

The written weights load into a state equal to the direct store's.
