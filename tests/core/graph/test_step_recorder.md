# tests/core/graph/test_step_recorder.cpp


Suites: `Small_StepRecorder`, `Small_StepRecorderOrder`, `Medium_StepRecordingDeathTest`,
`Medium_StepRecorderDeathTest`, `Medium_StepRecorderListDeathTest`.

Tests that StepRecorderList drives its members as one recorder, in the order added.

This file needs no device. `Graph::StepRecorder` and `Graph::StepRecorderList` are in
`modules/core/graph/include/SushiAI/graph/step_recorder.hpp`. The list is itself a `StepRecorder` and holds
references to its members; it does not own them. `add` returns a Status, and each case checks
its call with `SA_EXPECT_OK`. The members are `Graph::InMemoryStepRecorder`, which never reads
the recording it is given, so every case passes `Graph::StepRecording::detached()`. The cases
that observe the order across members build the members over one `InMemoryStepRecorder::Journal`.

## RecordOnlyRecorder

`RecordOnlyRecorder` overrides `record_into_step()` alone and returns 3. Its `before_replay()`
and `after_replay()` are the interface's own, which no in-memory adapter can show because it
overrides them.

## EmptyListRecordsNothing

An empty list reports `empty()` and returns 0 from `record_into_step()`.

## AddMakesTheListNonEmpty

After one `add` of an `InMemoryStepRecorder` the list is not empty and the recorder has received
no call, so `add` calls nothing on the member.

## RecordSumsTheMemberCounts

`InMemoryStepRecorder` members that report 2, 0 and 5 tasks give `record_into_step()` 7.

## RecordCallsMembersInTheOrderAdded

Rows: the members 0, 1, 2 added in the orders 0 1 2, 1 0 and 2 0 1. `record_into_step()` journals
a Record entry per member in the order added, not in the order of construction.

## BeforeReplayCallsMembersInTheOrderAdded

The same rows. `before_replay()` journals one BeforeReplay entry per member, in the order added.

## AfterReplayCallsMembersInTheOrderAdded

The same rows. `after_replay()` journals one AfterReplay entry per member, in the order added.
The members are not called in reverse.

## MemberWithoutReplayHooksIsDriven

A list of a `RecordOnlyRecorder` and an `InMemoryStepRecorder` of 4 tasks records 7. After both
replay hooks the adapter has received Record, BeforeReplay and AfterReplay, in that order.

## ListIsDrivenThroughTheInterface

Through a `StepRecorder&`, a list of two `InMemoryStepRecorder` members of 2 and 3 tasks records 5
tasks, and each member has received Record, BeforeReplay and AfterReplay, in that order.

## NestedListKeepsItsPlaceInTheOrder

An outer list of member 0, an inner list holding member 1, and member 2 records 7 tasks and
journals Record for 0, 1, 2 in that order.

## ADetachedRecordingHoldsNoRecorder

`StepRecording::detached()` reports `is_detached()`, and `tasks()` and `graph()` stop the process
with a report containing "detached" and the name of the recorder asked for. It is a death test.

## AddRejectsARecorderAlreadyHeld

A second `add` of the same recorder is a programmer's misuse, so it stops the process with a report
containing "StepRecorderList::add" and "already". The case runs as a death test in
`Medium_StepRecorderDeathTest`. The parent process then checks that the list still records that
member once: 2 tasks and one Record call.

## AddRejectsTheListItself

`list.add(list)` stops the process with a report containing "StepRecorderList::add" and "itself".
It is a death test in `Medium_StepRecorderListDeathTest`, which builds no fixture, and the list stays empty.
