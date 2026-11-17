# sample_models

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

Small models the tests build. `build_two_output_model` builds `Linear{6, 8}`, `GELU` and
`Linear{8, 3}` and publishes "features" and "logits". `build_extra_input_model` builds the sum of an
`[2, 2, 3]` input and an `[2, 2, 3]` extra input named "offset", then `Flatten` and `Linear{12, 3}`,
and publishes "logits"; the tests of an extra input read it.

## Depends on

`failure`, `shape` and `nn`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/sample_models/two_output_model.hpp` | `build_two_output_model` |
| `SushiAI/sample_models/extra_input_model.hpp` | `build_extra_input_model`, `extra_model_sample`, `K_EXTRA_INPUT_OFFSET`, `K_EXTRA_MODEL_ELEMENTS`, `K_EXTRA_MODEL_CLASSES` |
