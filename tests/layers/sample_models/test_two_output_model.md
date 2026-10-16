# tests/layers/sample_models/test_two_output_model.cpp


Suites: `Small_SampleModelsTwoOutput`.

Tests `build_two_output_model` in
`modules/layers/sample_models/include/SushiAI/sample_models/two_output_model.hpp`. Each case
traces the model in evaluation mode over three rows; no case needs a device.

## ATracePublishesTheFeaturesAndTheLogits

The trace publishes "features" then "logits", of the shapes [3, 8] and [3, 3].

## TheLinearLayersHoldTheFourParameters

The state walk yields four parameters of the shapes [8, 6], [8], [3, 8] and [3], so the GELU
between the two linear layers holds none.
