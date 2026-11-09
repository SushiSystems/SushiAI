# tests/layers/train/test_trainer/test_zones.cpp


Suites: `Medium_TrainerZones`.

Tests the zones and the period mark of `Trainer::fit` in
`modules/layers/train/include/SushiAI/train/trainer.hpp`. The fixture is an `EngineTest`. The case
trains a classifier of two linear layers over a synthetic dataset of two batches under a
SushiProfiler session that follows `train.step`, and reads the record the session returns.

## EachStepOfAFitWritesAPeriodWithItsGroups

One row per period, 0 and 1. `fit` runs two steps with `TimelineBinding` in place. The test then
marks a third period, which ends the second; a period without its end mark is not kept. The record
holds two periods, and the period of the row holds exactly the SushiAI zones `data.read`,
`host.feed`, `copy.input`, `device.run`, `host.loss` and `host.measure`, each in the group its name
starts with, and at least one zone of group `Runtime`, which SushiRuntime records while it runs
the step. The `train.fit` zone opens before the first period, so no period holds it.

The case exists only where `SA_PROFILE_LEVEL` is 1 or more, as the case in
`tests/core/profile/test_profile.cpp` does: a build that compiles every zone away has no period to
read, so the file holds no case there. The old case skipped itself in its body; the program's
build now decides.
