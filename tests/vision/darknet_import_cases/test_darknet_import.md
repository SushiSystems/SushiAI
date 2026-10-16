# tests/vision/darknet_import_cases/test_darknet_import.cpp


Suites: `Small_DarknetImport`, `Medium_DarknetImport`.

Tests `fill_imported_state` and `import_darknet_stem` in
`modules/vision/darknet_import_cases/include/SushiAI/vision/darknet_import_cases/darknet_import.hpp`.
The medium cases build a `NN::ModelState` from two entries on the shared runtime and fill it
from tensors made in memory, so no Darknet file is read. `check_imported_rows` and
`check_imported_weights` compare a real import against its translation; the suite
`DarknetImport` in `tests/vision/importers/test_darknet_import/test_import.cpp` runs them and
skips when the files of `SA_REFERENCE_DATA_DIR` are absent.

## two_entries

`two_entries` returns the float32 entries "0.weight" of [2, 3] and "0.bias" of [3].

## make_tensor

`make_tensor` returns an imported float32 tensor of a name and a shape whose element `i` holds a
base plus `i`.

## AStemWithoutFilesIsNotFound

`import_darknet_stem` of a stem no reference data directory holds fails with FileNotFound naming
its cfg.

## EveryTensorLandsInTheEntryOfItsName

The bias and the weight, given in the order opposite to the entries, land in the entries of
their names: the weight holds 0 to 5 and the bias 10 to 12. The payloads are copied, so they
compare exactly.

## ATensorWithoutAnEntryIsRefused

A tensor named "1.weight" fails with Incompatible, naming it and "no such entry".

## ARepeatedOrMisshapenTensorIsRefused

"0.bias" given twice fails with Incompatible naming "'0.bias' twice". "0.weight" of [3, 2]
against an entry of [2, 3] fails with Incompatible naming the tensor and its shape.

## AnEntryNoTensorSuppliesIsRefused

Tensors that supply "0.weight" alone fail with Incompatible, saying the import "does not
supply" "0.bias".
