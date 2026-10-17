# vision/darknet_forward_cases

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The forward pass of an imported Darknet model checked against what Darknet dumped: the
outputs of each layer, the heads and the detections after suppression.

## Depends on

- `failure`, `shape`, `allocation`, `umbrella`, `vision/detect`, `vision/importers`,
  `darknet_reference`, `engine_fixture`
  and `failure_expectations` (`PUBLIC_DEPENDS`). Of the core modules, `shape` and `allocation` are
  private (`PRIVATE_DEPENDS`).
- `config`, `graph`, `nn` and `train` (`PRIVATE_DEPENDS`).

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/vision/darknet_forward_cases/darknet_forward.hpp` | The forward cases; see the header |
