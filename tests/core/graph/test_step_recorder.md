# tests/core/graph/test_step_recorder.cpp


Suites: `Medium_StepRecorder`, `Medium_StepRecorderDeathTest`.

Tests that StepRecorderList drives its members as one recorder, in the order added.

This file needs no device. `Graph::StepRecorder` and `Graph::StepRecorderList` are in
`modules/core/graph/include/SushiAI/graph/step_recorder.hpp`. The list is itself a `StepRecorder` and holds
references to its members; it does not own them. `add` returns a Status, and each case checks
its call with `SA_EXPECT_OK`.

## JournalRecorder

`JournalRecorder` appends "<name>.record", "<name>.before" or "<name>.after" to a journal the
test owns, and `record_into_step()` returns the task count given at construction.

## RecordOnlyRecorder

`RecordOnlyRecorder` overrides `record_into_step()` alone and returns 3. Its `before_replay()`
and `after_replay()` are the interface's own.

## in_order_added

`in_order_added` returns the three journal entries that recorders named a, b and c write for
one call.

## EmptyListRecordsNothing

An empty list reports `empty()`, returns 0 from `record_into_step()` and does not throw from
either replay hook.

## AddMakesTheListNonEmpty

After one `add` the list is not empty and the journal is still empty, so `add` calls nothing
on the member.

## RecordSumsTheMemberCounts

Members that report 2, 0 and 5 tasks give `record_into_step()` 7.

## RecordCallsMembersInTheOrderAdded

`record_into_step()` over a, b and c journals "a.record", "b.record", "c.record".

## OrderFollowsAddNotConstruction

With b added before a, the journal is "b.record", "a.record", although a was constructed
first.

## BeforeReplayCallsMembersInTheOrderAdded

`before_replay()` over a, b and c journals "a.before", "b.before", "c.before".

## AfterReplayCallsMembersInTheOrderAdded

`after_replay()` over a, b and c journals "a.after", "b.after", "c.after". The members are not
called in reverse.

## MemberWithoutReplayHooksIsDriven

A list of a `RecordOnlyRecorder` and a journal recorder of 4 tasks records 7. After both replay
hooks the journal holds "a.record", "a.before" and "a.after" only.

## ListIsDrivenThroughTheInterface

Through an `StepRecorder*`, a list of a and b records 5 tasks and journals the record calls,
then the before calls, then the after calls, each as a then b.

## NestedListKeepsItsPlaceInTheOrder

An outer list of a, an inner list holding b, and c records 7 tasks and journals "a.record",
"b.record", "c.record".

## AddRejectsARecorderAlreadyHeld

A second `add` of the same recorder is a programmer's misuse, so it stops the process with a report
containing "StepRecorderList::add" and "already". The case runs as a death test in
`Medium_StepRecorderDeathTest`. The parent process then checks that the list still records that
member once: 2 tasks and one journal entry.

## AddRejectsTheListItself

`list.add(list)` stops the process with a report containing "StepRecorderList::add" and "itself".
It is a death test in `Medium_StepRecorderDeathTest`, and the list stays empty.
