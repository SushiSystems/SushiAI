# tests/layers/sample_models/test_extra_input_model.cpp

Suites: `Small_SampleModelsExtraInput`.

Tests `build_extra_input_model` and the extents of
`modules/layers/sample_models/include/SushiAI/sample_models/extra_input_model.hpp`. No case traces
the model or needs a device. The trace of the model, with its leaf and its loss, is checked through
the trainer in `tests/layers/train/test_extra_inputs.cpp`.

## TheModelDeclaresOneExtraInputOfTheSampleShape

The graph lists one extra input named `offset`, of the per-sample shape [2, 2, 3] and element type
float32.

## TheExtraInputIsFoundByItsName

`extra_input("offset")` succeeds; `extra_input("absent")` fails with `InvalidDescription` naming
`absent`.

## TheElementCountIsTheElementCountOfTheSample

`K_EXTRA_MODEL_ELEMENTS`, the width the model's linear layer reads, equals the element count of
`extra_model_sample()`.
