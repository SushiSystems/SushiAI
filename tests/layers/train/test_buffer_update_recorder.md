# tests/layers/train/test_buffer_update_recorder.cpp


Suites: `Small_BufferUpdateRecorder`, `Small_BufferUpdateRecorderCount`, `Small_BufferUpdateBlend`.

The recorder blends each buffer's batch statistic into its running buffer, once per replay. The
cases check how many tasks it records, the value one replay leaves behind, and the two refusals
that binding makes. Every suite runs on the engine of a `HostExecutionBackend`, and a replay is
the replay of the `StepTasks` that backend hands out.

## HostBlend

`HostBlend` owns the `HostExecutionBackend` and `replay_once`, which replays a `StepTasks` once with
a report of its own. The three fixtures derive from it.

## SlotCountRow

`SlotCountRow` holds a slot count and the number of tasks a recorder of that many slots records.

## BlendRow

`BlendRow` holds one running element and the batch element blended into it.

## TwoTasksPerSlotAndNoneForNoSlot

Each row builds a recorder over that many slots of two float64 elements and records it into the
tasks of the backend. `slot_count()` is the row's count and `record_into_step` returns twice it:
0 for no slot, 2 for one, 4 for two.

## OneReplayBlendsTheSourceIntoTheTarget

One slot over one float64 element. After one replay the element equals the
`NN::updated_statistic` of its running value and its batch value, with momentum 0.1 and a source
scale of 2, within four rounded operations of double. The rows are the pairs (1, 0.25), (-2, 3),
(0.5, -1) and (4, 2).

## BindRefusesATargetTheStateLacks

The state holds one entry, `bn.running_mean`, under identity 1. The update targets identity 7.
Binding is refused as `Incompatible`, and the message contains "no entry".

## BindRefusesADTypeMismatch

The target entry is float32 and the source tensor is float64. Binding is refused as
`Incompatible`, and the message names both dtypes by their `dtype_name`.
