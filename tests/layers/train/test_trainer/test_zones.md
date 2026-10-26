# tests/layers/train/test_trainer/test_zones.cpp


Suites: `Medium_TrainerZones`.

Tests the zones and the period mark of `Trainer::fit` in
`modules/layers/train/include/SushiAI/train/trainer.hpp`. The fixture is an `EngineTest`. The case
trains a classifier of two linear layers over a synthetic dataset of two batches under a
SushiProfiler session that follows `train.step`, and reads the record the session returns.

## AFitOfTwoStepsWritesTwoPeriodsWithTheirGroups

`fit` runs two steps with `TimelineBinding` in place. The test then marks a third period, which
ends the second; a period without its end mark is not kept. The record holds two periods, and
each holds exactly the zones `data.read`, `host.feed`, `copy.input`, `device.run`, `host.loss`
and `host.measure`, each in the group its name starts with. The case skips itself when
`SA_PROFILE_LEVEL` is 0. The `train.fit` zone opens before the first period, so no period holds
it.
