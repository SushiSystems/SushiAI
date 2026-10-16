# tests/core/graph/test_replay_prelude.cpp


Suites: `Small_ReplayPrelude`.

Tests `ReplayPrelude`, an alias of a callable that takes the count of finished replays. The
header declares nothing else. A lowering row hands a prelude to its step through
`LoweringContext::before_each_replay`, which `tests/core/graph/lowering/test_context.cpp` tests.

## ADefaultPreludeHoldsNoTarget

A default prelude converts to false. A context refuses such a prelude.

## APreludeReceivesTheCountOfFinishedReplays

A prelude called with 0, 1 and 2 receives those counts in that order.
