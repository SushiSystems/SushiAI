# tests/vision/darknet_import_cases/test_darknet_import.cpp


Suites: `Medium_DarknetImportFiles`, `Medium_DarknetImportCases`, `Small_DarknetImportEntries`.

Tests `check_tensors_against_entries`, `fill_imported_state` and `import_darknet_stem` in
`modules/vision/darknet_import_cases/include/SushiAI/vision/darknet_import_cases/darknet_import.hpp`.
The refusal cases call `check_tensors_against_entries` on the two entries and tensors made in
memory, so they need no engine and no Darknet file. The medium fill case builds a `NN::ModelState`
from the two entries on the shared runtime and fills it from tensors made in memory. `check_imported_rows` and
`check_imported_weights` compare a real import against its translation, and
`check_written_import` compares the files a real import writes against it; the suite
`DarknetImport` in `tests/vision/importers/test_darknet_import/test_import.cpp` runs them and
skips when the files of `SA_REFERENCE_DATA_DIR` are absent.

## two_entries

`two_entries` returns the float32 entries "0.weight" of [2, 3] and "0.bias" of [3].

## make_tensor

`make_tensor` returns an imported float32 tensor of a name and a shape whose element `i` holds a
base plus `i`.

## AStemWithoutFilesIsNotFound

`Medium_`, since the import looks for files. `import_darknet_stem` of a stem no reference data directory holds fails with FileNotFound naming
its cfg.

## EveryTensorLandsInTheEntryOfItsName

The bias and the weight, given in the order opposite to the entries, land in the entries of
their names: the weight holds 0 to 5 and the bias 10 to 12. The payloads are copied, so they
compare exactly.

## ATensorWithoutAnEntryIsRefused

A tensor named "1.weight" fails with Incompatible, naming it and "no such entry".

## ATensorStatedTwiceIsRefused

"0.bias" given twice fails with Incompatible naming "'0.bias' twice".

## ATensorOfAnotherShapeIsRefused

"0.weight" of [3, 2] against an entry of [2, 3] fails with Incompatible naming the tensor and
its shape.

## AnEntryNoTensorSuppliesIsRefused

Tensors that supply "0.weight" alone fail with Incompatible, saying the import "does not
supply" "0.bias".
