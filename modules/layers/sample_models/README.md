# sample_models

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

Small models the tests build. `build_two_output_model` builds `Linear{6, 8}`, `GELU` and
`Linear{8, 3}` and publishes "features" and "logits".

## Depends on

`core` and `nn`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/sample_models/two_output_model.hpp` | `build_two_output_model` |
