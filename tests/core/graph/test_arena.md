# tests/core/graph/test_arena.cpp


Suites: `Integration_Arena`, `Integration_ArenaDeathTest`.

The memory planner rests on three device-side claims, and the file checks each one.

1. Two values packed into one arena report non-overlapping `SushiRuntime::BufferRegion`s over the same
   allocation base. Packing is safe only because of this. SushiRuntime's dependency tracker
   orders two accesses if and only if their regions overlap. Every driver once passed a region
   that named the bare allocation base; with that region every tensor in an arena would alias
   every other one, and the tracker would serialise them all.
2. A view addresses its own elements. `storage_offset` counts elements, so an offset that is
   not a whole number of them is refused. Truncating it would give a tensor that starts in the
   wrong place without an error.
3. Mixed dtypes in one arena each start aligned for their own element width. This is the one
   part of the offset arithmetic `Tensor` cannot do for the planner, because the offset is
   already chosen by the time a `Tensor` exists.

The file is laid out in four groups: region precision, element-counted offsets, mixed dtypes in
one arena, and the pool's side of the contract.

## TwoPackedTensorsShareABaseButNotAByte

The plan packs both values of the chain at different offsets, and the `ArenaSet` holds one arena.
The two `[16]` float32 views report regions with the same base, which is the allocation base of the
storage, and neither region overlaps the other. Each region is 64 bytes long and starts at the
offset its placement states.

## OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther

This is the functional half of the region claim. Real SushiBLAS work runs on two views of one
arena in one step, and each result must land only in its own bytes.

## AViewAddressesItsOwnElements

The view of the second value has a `storage_offset` equal to its placement offset divided by the
size of a float. Its data pointer lies exactly that many bytes past the allocation base.

## AnOffsetThatIsNotAWholeNumberOfElementsIsRefused

An offset of four bytes into a float64 view can be written in bytes but not in the element
count `Tensor` stores. Truncating it would put the tensor somewhere the planner never reserved.
The same offset is valid for float32.

## AViewThatWouldRunPastTheArenaIsRefused

`view_tensor` of 64 float32 elements at byte 64 of a 128 byte arena is a fatal stop whose message
contains "run past the end".

## MixedDtypesEachStartAlignedForTheirOwnElementWidth

The plan's alignment is set to 1 so that only the element-width floor holds the two values
apart. With the default alignment the property holds for an uninteresting reason; at 1 it
holds only because the planner rounds.

The round trip through each view proves the addresses are usable as well as aligned. A float64
read through a mis-rounded start would come back as two halves of neighbouring values, not as
the number written.

## AdoptingASecondPlanIsRefused

A `TensorPool` adopts one plan. Adopting a second is a fatal stop whose message contains "already
adopted".

## TheEmptyPlanPacksNothing

The empty plan is the control an A/B measurement of the whole pass needs. It is also what a
caller uses to get unpacked allocation.

## APlannedPoolAllocatesOneStorageForAWholeArena

The test compares no byte counts. Its tensors are 64 bytes each and the arena aligns every slot
to 128, so packing them costs memory. The saving belongs to realistic tensor sizes and is
measured where that holds, in `Regression_StaticMemory`.
