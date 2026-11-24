# tests/layers/io/test_exporter_registry.cpp

Suites: `Small_ExporterRegistry`.

Tests `IO::ExporterRegistry` and `IO::export_model`, in `modules/layers/io/include/SushiAI/io/exporter_registry.hpp`. Two rows defined in the test stand for the exporters; none of the real ones is linked.

## AnAddedRowIsFoundByName

Two rows are added. `rows()` holds them in insertion order, and `find` returns the row of a known format and a null pointer for an unknown one.

## ASecondRowOfOneFormatIsRefused

A row added twice fails with `InvalidDescription`, naming the format and saying it is already registered.

## TheWeightWordIsRefused

A row named `weights` fails. The word belongs to `export weights`, so no format may take it.

## ExportModelCallsTheRowOfTheFormat

`export_model` calls the row of the format with the model and the stem. The file it returns carries the stem and a byte that counts the model's notes.

## AnUnknownFormatListsTheFormatsTheRegistryHolds

An unknown format fails with `InvalidDescription`, naming it and listing the formats the registry holds, in order.
