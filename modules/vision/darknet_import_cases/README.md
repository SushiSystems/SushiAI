# vision/darknet_import_cases

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The import of a Darknet model checked against its dump: the model file it writes, the
weights it stores and the model it rebuilds.

## Depends on

- `failure`, `dtype`, `shape`, `io`, `nn`, `umbrella`, `vision/importers`, `engine_fixture` and
  `failure_expectations`
  (`PUBLIC_DEPENDS`). Of the core modules, `dtype` and `shape` are private (`PRIVATE_DEPENDS`).
- `config`, `train`, `vision/detect`, `vision/darknet_forward_cases` and `darknet_reference`
  (`PRIVATE_DEPENDS`).

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/vision/darknet_import_cases/darknet_import.hpp` | The import cases; see the header |
