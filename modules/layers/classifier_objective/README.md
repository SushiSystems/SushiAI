# classifier_objective

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

`classifier_objective`, the cross-entropy objective of a graph `Train::build_classifier()`
traced, read from the extents of its target leaf.

## Depends on

`graph` and `train`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/classifier_objective/classifier_objective.hpp` | `classifier_objective` |
