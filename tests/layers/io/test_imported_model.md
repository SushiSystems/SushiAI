# tests/layers/io/test_imported_model.cpp


Suites: `Small_ImportedModel`.

Tests the structs `IO::ImportedModel` and `IO::ImportedTensor` of
`modules/layers/io/include/SushiAI/io/imported_model.hpp`. The header holds data only, so the
cases pin the empty state an importer starts from and the payload size a tensor carries. What the
writer does with an imported model is tested in `test_imported_model_writer.cpp`.

## ADefaultModelStatesNothing

A default `ImportedModel` has no weights, no notes, no layers, and a null `training` and a null
task.

## ATensorHoldsTheBytesItsRecordStates

A float32 record of [4, 3] sizes the payload to 48 bytes, and the same record as int64 states 96.
