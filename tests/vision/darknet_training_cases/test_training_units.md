# tests/vision/darknet_training_cases/test_training_units.cpp


Suite: `Small_TrainingUnits`.

Tests `units` in
`modules/vision/darknet_training_cases/include/SushiAI/vision/darknet_training_cases/training_units.hpp`,
which the six Darknet training suites use to state a bound as a multiple of the float unit roundoff u.

## OneUnitIsTwoToTheMinus24

`units(1)` is 2^-24, `units(0)` is 0 and `units(8388608)`, which is 2^23 units, is 0.5. Each value is
an exact power of two, so the comparison is equality.
