# tests/applications/programs/source/test_weights_export.cpp

Suites: `Small_WeightsExport`.

Tests `compose_weight_writers` and `export_weights_file` of
`applications/programs/source/weights_export.hpp`. The cases write nothing but an empty scratch
folder. A full export of a fixture's weights is in `test_model_export.cpp`.

## TheComposedWritersAreSafetensorsThenTorch

The writers are `safetensors` and `torch`, in that order, and each has a function.

## AnUnknownFormatNamesTheFormatsAndWritesNoFile

The format `npz` fails with `InvalidDescription` naming `'npz'`, `'safetensors'` and `'torch'`, and
the output file does not exist afterwards.

## AnExportOfAMissingModelFileFailsAndWritesNoFile

A request whose model and weight files do not exist fails, and the output file does not exist
afterwards.
