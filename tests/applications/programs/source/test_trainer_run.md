# tests/applications/programs/source/test_trainer_run.cpp


Suites: `Small_TrainerRun`, in `sushiai_programs_tests`.

Tests `register_runs`, which sets the fallback run of a `RunRegistry`. A run through that
fallback is covered by `AClassificationRunTrainsThroughTheFallback` in `test_training_run.cpp`;
these cases touch only the registry and need no device.

## RegisteringSetsTheFallbackAndAddsNoRow

After `register_runs`, `fallback()` is not null and `rows()` is empty.

## ASecondRegistrationIsRefused

A second `register_runs` on the same registry fails with `StatusCode::InvalidDescription`, and
`fallback()` still returns the run the first call set.
