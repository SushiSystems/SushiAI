# vision/darknet_import_cases

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The import of a Darknet model checked against its dump: the model file it writes, the
weights it stores and the model it rebuilds.

## Depends on

`umbrella`, `vision/detect`, `vision/importers`, `vision/darknet_forward_cases`, `darknet_reference`, `engine_fixture`, `failure_expectations` and `tensor_values`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/vision/darknet_import_cases/darknet_import.hpp` | The import cases; see the header |
