# tests/core/eager/test_trailing_ledger.cpp


Suites: `Small_TrailingLedger`.

Tests which trailing work `TrailingLedger` plans for a step and what it remembers.

This file needs no device. A case builds a `StepWork` with region keys and an
`InMemoryStepRecorder` as the trailing work, and asks the ledger for a plan.
`modules/core/eager/README.md`, under `trailing_ledger.hpp`, gives the two decisions of a plan.

## AStepWithNoTrailingWorkPlansNothing, NewTrailingWorkIsPlannedForRecordingAndRetiresNothing

A fresh ledger retires nothing and records only what the step hands it.

## HeldTrailingWorkUnderTheSameKeyIsKept

The recorder and key the ledger holds are neither retired nor recorded, which is why the trailing
work is recorded once across steps.

## HeldTrailingWorkUnderAMovedKeyIsRetiredAndRecordedAgain, AnotherRecorderRetiresTheHeldOneAndIsRecorded, AStepWithoutTrailingWorkRetiresTheHeldOne, AFailedRecordingIsRetiredAndRecordedAgain

The four ways held trailing work leaves: its key moved, another recorder took over, the step has
none, or its recording began and did not finish.

## TrailingWorkWithNoKeyIsRefused, ATrailingKeyEqualToARegionKeyIsRefused, ATrailingKeyBelowARegionKeyIsRefused

The trailing work runs behind every region only when its key lies above theirs, so any other key is
refused with `InvalidDescription`.

## HeldTrailingWorkIsNotCheckedAgainstTheRegionKeys

A plan that records nothing does not look at the keys.

## EachBegunRecordingIsCounted, ItReportsTheTasksOfTheHeldRecording, ClearForgetsTheKeyAndTheTasks

What the ledger remembers: the number of recordings begun, the task count and the key of the held
recording, and nothing after `clear`.

## TheReplayHooksReachTheHeldRecorder, TheReplayHooksDoNothingWithoutAHeldRecorder

`before_replay` and `after_replay` reach the recorder the ledger holds, once per call, and stop at
a ledger that holds none.
