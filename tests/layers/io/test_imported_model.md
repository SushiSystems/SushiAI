# tests/layers/io/test_imported_model.cpp


Suites: `Small_ImportedModel`.

Tests the payload size a tensor of `IO::ImportedTensor` states, in
`modules/layers/io/include/SushiAI/io/imported_model.hpp`. The header holds data only, so the one
promise worth a test is the payload size its record states. What the writer does with an imported
model is tested in `test_imported_model_writer.cpp`.

## PayloadRow

Names an element type and the bytes twelve elements of it take.

## Small_ImportedModel

Runs `ATensorHoldsTheBytesItsRecordStates` once for each row.

## ATensorHoldsTheBytesItsRecordStates

A record of [4, 3] states a payload of its element count times its element size: 48 bytes as
float32 and 96 as int64.
