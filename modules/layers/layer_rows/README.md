# layer_rows

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The rows of the layer registry a test builds: `make_layer_registry` returns a registry that
holds what `register_layers` adds, `types_of` lists the type of each row, and `MinimalCase` with
`make_minimal` builds the smallest object of one row over its input shapes.

## Depends on

`failure`, `shape`, `nn`, `model_text` and `failure_expectations`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/layer_rows/layer_registry.hpp` | `make_layer_registry`, `types_of` |
| `SushiAI/layer_rows/minimal_case.hpp` | `MinimalCase`, `span_of`, `make_minimal` |
