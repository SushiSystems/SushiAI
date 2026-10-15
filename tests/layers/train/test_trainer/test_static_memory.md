# tests/layers/train/test_trainer/test_static_memory.cpp


Suites: `Regression_StaticMemory`.

The file tests four guarantees of the static memory planner whose failure would be silent: the loop
still converges, the run still reports success, and nobody notices.

1. A training loop allocates nothing after the first step. The test counts this. Every Storage
   allocates through the runtime, so `AdvancedControls::live_allocation_count()` is the
   live-allocation count, and one extra buffer moves it by exactly one.
2. Packing shrinks the footprint. A pass that quietly stopped packing would leave every other test
   in the suite green. The footprint is measured on the unfused graph, because fusion reaches the
   same bytes by a different route: it deletes the intermediates where packing overlaps them. A
   planner test that let fusion do its work first would be measuring the catalog.
3. Packing changes no number. If it does, two values are aliasing that should not be, and the right
   response is to find them, not to record the reference again.
4. A buffer the planner packed is rewritten every replay. A K_ZERO buffer that is filled once and
   then only read holds zeros for exactly one step and then whatever overwrote it. Such a model
   trains, badly, for ever.

## train

With `packed` false, the pool has already adopted the empty plan, which covers no value and asks
for no arena. That is the behaviour the library had before the planner pass existed, which makes it
the only honest control.

The `fusion` parameter says what the fusion pass is allowed to do. Fusion is a confound in these
measurements: a fused subtree's intermediates stop being values, so fusion removes exactly the
buffers packing would overlap. Every test in this file therefore says which graph its claim is
about.

## ATrainingLoopAllocatesNothingAfterTheFirstStep

This is the headline property of the planner pass, and the test counts it. The baseline is taken
after the first step, so a one-time lazy cache inside the runtime is tolerated while a per-step
allocation is not. A per-step buffer would still train correctly and would show up only as a slowly
growing process.

## PackingShrinksTheFootprintOfTheSameModel

The footprint is measured on the unfused graph on purpose. Fusion is a different mechanism reaching
the same bytes: a fused subtree's intermediates stop being values, so they need no slot where
packing would give them a shared one. Planning what fusion already deleted would make this test a
statement about the fusion catalog and not about the planner, and it would stop testing anything
the day a new catalog entry landed.

## FusingTakesTheBuffersPackingUsedToOverlap

This test states the interaction between fusion and packing so that nobody has to rediscover it. On
the fused graph the planner's own report reads like a broken planner: every remaining activation's
live range intersects every other's, so the planner overlaps nothing and prints "0% saved". The
planner is not broken. The buffers it would overlap do not exist, because a fused subtree's
intermediates are not values any more. That is the better outcome of the two, and it is why the
arena has nothing left to do.

`unpacked_bytes` measures this directly. It is what the same tensors would cost at one allocation
each, so a drop in it means buffers disappeared and were not merely shared.

Against the packed-but-unfused graph the fused footprint is never worse. On this model it is
exactly equal, because what fusion deletes here happens to be what the planner was overlapping. The
assertion is written as `<=` and not `==` on purpose: the equality is a property of this model's
shape and not of the two passes, and pinning it would fail on the next catalog entry for a reason
nobody should have to investigate.

## PackedAndUnpackedTrainingAgreeByteForByte

If the two runs ever differ, two tensors are sharing bytes they should not. The fix is to find
which two, never to relax the comparison. The test compares bytes and not a tolerance for the
reason test_determinism.cpp gives: a tolerance cannot tell a rounding difference from a real
overlap.

## APackedZeroBufferIsRefilledOnEveryReplay

K_ZERO is the one operation whose output nothing else writes. It would work perfectly if it were
filled once outside the step, until the planner handed its bytes to a later value. The test
corrupts the buffer from the host between two replays, which is what a reused arena slot does, and
the second replay must still see zeros.
