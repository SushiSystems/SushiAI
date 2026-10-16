# tests/core/graph/test_tensor_pool.cpp


Suites: `Medium_TensorPool`.

Tests how `TensorPool` hands out, shares and counts tensors when no memory plan is adopted. A
pool with a plan is tested in `tests/core/graph/test_arena.cpp`. Every case allocates on an
engine. Constant fills and copies are compared exactly, since float32 holds every value used.

## AValueIsAllocatedOnceAndKeepsItsTensor

`materialise` of a [4, 6] value twice returns one tensor, which `tensor` returns too. The pool
holds one tensor, books 96 bytes under `values`, and its allocated and unpacked bytes are equal.

## OnesAreSharedPerShapeAndTypeAndFilledByTheFlush

`ones` for a float32 [3] returns the same tensor twice, and a float64 [3] returns another. The
pool books the bytes of both under `ones`. After `flush_constants` every element of both is 1.

## ScratchIsKeyedByNodeAndSlot

`scratch` for node 4, slot 0 returns the same tensor twice. Slot 1 of node 4 and slot 0 of node 5
return other tensors. The pool holds three tensors and books 48 bytes under `scratch`.

## AnAliasSharesTheStorageOfItsSource

A tensor from `create` is bound to one value and a second value aliases it, twice, which is
accepted. Values written through the first are read back through the second, and both have one
storage.

## AQueuedConstantIsWrittenByTheFlush

A flush with nothing queued succeeds. After `request_constant` of 2.5 on a created [4] tensor,
the flush writes 2.5 to every element.
