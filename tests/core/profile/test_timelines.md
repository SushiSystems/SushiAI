# tests/core/profile/test_timelines.cpp

Suites: `Small_Timelines`, `Medium_Timelines`.

Tests `Timeline`, `mark_period` and `timeline_name` in
`modules/core/profile/include/SushiAI/profile/timelines.hpp`, and
`TimelineBinding`. The cases start a real SushiProfiler session that follows
`train.step` for one warm-up period and three active ones, and stop it before they end.

## NamesAreTrainStepAndInferenceBatch

`timeline_name` returns `train.step` for `TrainStep` and `inference.batch` for
`InferenceBatch`.

## MarkWithNoBindingDoesNothing

With a session running and no `TimelineBinding`, `mark_period` on either timeline leaves the
session's `train.step` timeline at `K_NO_PERIOD`.

## ABoundSessionRecordsOnePeriodPerMark

Five `TrainStep` marks under a binding open periods 0 to 4 of the session's `train.step`; the
record holds four periods, the first a warm-up, with indices 0 to 3. Two `InferenceBatch`
marks move `inference.batch` to period 1. A mark after the binding is gone changes nothing.
