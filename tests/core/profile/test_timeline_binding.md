# tests/core/profile/test_timeline_binding.cpp

Suites: `Medium_TimelineBinding`.

Tests `TimelineBinding` in `modules/core/profile/include/SushiAI/profile/timeline_binding.hpp`
against a real SushiProfiler session that follows `train.step`.

## MarksStopWhenTheBindingEndsAndResumeWithTheNext

A mark under a binding opens period 0. After the binding ends, a mark leaves the timeline at
period 0. A second binding on the same session marks period 1.
