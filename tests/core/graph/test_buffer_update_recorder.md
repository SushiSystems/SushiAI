# tests/core/graph/test_buffer_update_recorder.cpp

**Describes:** `tests/core/graph/test_buffer_update_recorder.cpp`, `modules/core/graph/include/SushiAI/graph/buffer_update_recorder.hpp`

Suites: `Small_BufferUpdateRecorderCount`, `Small_BufferUpdateBlend`.

The recorder blends each buffer's batch statistic into its running buffer, once per replay. The
cases check how many tasks it records and the value one replay leaves behind. Every suite runs on
the engine of a `HostExecutionBackend`, and a replay is the replay of the `StepTasks` that backend
hands out.

## HostBlend

`HostBlend` owns the `HostExecutionBackend` and `replay_once`, which replays a `StepTasks` once with
a report of its own. The two fixtures derive from it.

## SlotCountRow

`SlotCountRow` holds a slot count and the number of tasks a recorder of that many slots records.

## BlendRow

`BlendRow` holds one running element and the batch element blended into it.

## TwoTasksPerSlotAndNoneForNoSlot

Each row builds a recorder over that many slots of two float64 elements and records it into the
tasks of the backend. `slot_count()` is the row's count and `record_into_step` returns twice it:
0 for no slot, 2 for one, 4 for two.

## OneReplayBlendsTheSourceIntoTheTarget

One slot over one float64 element. After one replay the element equals
`(1 - momentum) * running + momentum * source_scale * batch`, with momentum 0.1 and a source scale
of 2, within three rounded operations of double: the scale rounds 1 - momentum and the product, the
axpy rounds momentum * scale and the product, and one sum joins the two. The rows are the pairs
(1, 0.25), (-2, 3), (0.5, -1) and (4, 2).
