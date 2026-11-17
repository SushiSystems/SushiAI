# tests/layers/train/test_leaf_writes.cpp

Suites: `Small_LeafWrites`, `Small_LeafWritesType` (one row per element type).

Tests `Train::mark_leaves_unwritten` and `Train::require_leaves_written`, the pair `Trainer::run_step`
puts around `objective.feed` (RW-111). The leaves are inputs of a graph and their tensors are
allocated on the engine of a `HostExecutionBackend` and bound in a `TensorPool`, so no case needs
a device.

## ALeafNoFeedWroteIsRefused

One row per element type: float32, float64 and int32. A leaf of three elements is marked and not
written. `require_leaves_written` fails with `InvalidDescription`, "'targets'", "element 0" and
"unwritten". The
pattern repeats four bytes, so a type whose element is 4 or 8 bytes is covered by both lengths.

## AStaleValueIsNotMistakenForAWrite

The leaf holds 7, 7, 7 from an earlier step. Marking overwrites it, so a feed that skips the leaf is
refused: without the mark the stale 7s would read as written.

## ALeafAFeedWroteIsAccepted

After the mark, the leaf is written 1, 0, 0. The check succeeds.

## OnlyTheSkippedLeafIsNamed

Two leaves, `targets` and `weights`, are marked and every element of `targets` is written. The
refusal names 'weights' and its element 0, and does not name 'targets'.

## ALeafWrittenInPartIsRefusedNamingTheElement

A float32 leaf of three elements is marked and its elements 0 and 1 are written. The check fails
with `InvalidDescription`, "'targets'", "element 2" and "unwritten": one element still holding the
pattern is enough to refuse the leaf, and the message gives the first such index.

## ALeafThePoolDoesNotBindIsSkipped

A leaf declared and not bound in the pool is neither marked nor refused: no feed could have written
a tensor that does not exist, and the objective's own `find_leaf_tensor` reports that case.
