# tests/core/profile/test_timelines.cpp

Suites: `Small_Timelines`, `Medium_Timelines`. A case that starts a session is `Medium_`, since
the session runs a consumer thread.

Tests `Timeline` and `mark_period` in
`modules/core/profile/include/SushiAI/profile/timelines.hpp`, and
`TimelineBinding`. The cases start a real SushiProfiler session that follows
`train.step` for one warm-up period and three active ones, and stop it before they end.

## NamesAreTrainStepAndInferenceBatch

`timeline_name` gives `train.step` for `TrainStep` and `inference.batch` for `InferenceBatch`, the
names a profile configuration follows. The case starts no session.

## MarkWithNoBindingDoesNothing

`mark_period` of either timeline, with no `TimelineBinding` in force, leaves the `train.step` and
`inference.batch` timelines of a running session at `K_NO_PERIOD`.

## MarkOnANullSessionDoesNothing

`mark_period` on either timeline with a null session leaves the `train.step` timeline of a
running session at `K_NO_PERIOD`. The case reads no `TimelineBinding`.

## AMarkOnASessionOpensAPeriodOfItsTimelineOfThatName

`mark_period(Timeline::TrainStep, session)` moves the session's `train.step` timeline to period
0 and leaves its `inference.batch` timeline at `K_NO_PERIOD`. No binding is made.

## ABoundSessionRecordsOnePeriodPerMark

Five `TrainStep` marks under a binding open periods 0 to 4 of the session's `train.step`; the
record holds four periods, the first a warm-up, with indices 0 to 3. Two `InferenceBatch`
marks move `inference.batch` to period 1. A mark after the binding is gone changes nothing.
