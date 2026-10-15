# vision/darknet_training_cases

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The training step of a Darknet model checked against its dump: the deltas of the heads,
the updates of the convolutions, the rolling statistics, the memory plan of the step and the
one-batch overfit.

## Depends on

- `config`, `core`, `umbrella`, `vision/detect`, `darknet_reference`, `engine_fixture` and
  `failure_expectations` (`PUBLIC_DEPENDS`).
- `graph`, `nn`, `optim`, `train`, `vision/importers`, `vision/training` and
  `vision/darknet_forward_cases` (`PRIVATE_DEPENDS`).

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/vision/darknet_training_cases/darknet_training.hpp` | The training cases; see the header |
