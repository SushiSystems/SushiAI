# tests/core/eager/test_trailing_chain.cpp

Suites: `Small_TrailingChain`.

Tests `TrailingChain` of `trailing_chain.hpp`, which runs a step's copies, the caller's recorder
and the blends as one recorder. Every case runs on a `HostExecutionBackend`, so no device is
needed. The bank holds four two-element FLOAT64 tensors: the gradient (1, 2), the copy target
(0, 0), the running statistic (10, 10) and the batch value (2, 4). The spec copies the gradient into
the copy target and blends the batch value into the statistic with the weight one half.

## RecordingReportsTheCopiesTheCallersTasksAndTheBlends

A scripted recorder that reports five tasks sits between the copy and the blend. The chain reports
one task for the copy, the five of the caller and two for the blend, and the caller's recorder
received one `Record` call.

## OneReplayCopiesTheGradientAndBlendsTheBatchValue

After the chain is recorded, sealed and replayed once, the copy target holds (1, 2) and the
statistic holds (1 - 0.5) * 10 + 0.5 * 2 = 6 and (1 - 0.5) * 10 + 0.5 * 4 = 7. Every product is a
power of two times an integer, so the values are exact and compared with `==`.

## ASpecOfNoCopyAndNoBlendRecordsOnlyTheCallersTasks

With no copy and no blend the chain reports the five tasks of the caller and no more.

## TheFailureOfTheCallersRecorderIsTheFailureOfTheChain

A recorder scripted to fail with `InvalidDescription` and a message makes `record_into_step` of the chain
fail with the same code and message.

## TheReplayHooksReachTheCallersRecorder

`before_replay` then `after_replay` of the chain reach the caller's recorder in that order, once
each.
