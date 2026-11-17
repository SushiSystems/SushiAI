# tests/applications/programs/source/test_weights_import.cpp

Suites: `Small_WeightsImport`.

Tests `compose_weight_sources` and `import_weights_file` of
`applications/programs/source/weights_import.hpp`. The cases write nothing but an empty scratch
folder. A full import of a weight file into a model's records, with its name map, is covered where the
pieces are: `tests/layers/io/test_weight_import.cpp` for the fill and the message, and
`tests/layers/io/test_weight_source.cpp` for the two containers.

## TheComposedSourcesAreNpzThenPytorchThenSafetensors

The registry holds the rows `npz`, `torch` and `safetensors`, in that order.

## AnImportOfAMissingModelFileFailsAndWritesNoFile

A request whose model and weight files do not exist fails, and the output file does not exist
afterwards.
