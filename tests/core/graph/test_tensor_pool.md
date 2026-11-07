# tests/core/graph/test_tensor_pool.cpp


Suites: `Small_TensorPool`, `Medium_TensorPool`, `Medium_TensorPoolDeathTest`.

Tests how `TensorPool` hands out, shares and counts tensors when no memory plan is adopted. A
pool with a plan is tested in `tests/core/graph/test_arena.cpp`. The `Small_` cases build the pool
over a `Graph::InMemoryTensorSource`, which describes each tensor and gives it no storage, and
read the bookkeeping: which tensor comes back, how many were requested and how many bytes are
booked. The `Medium_` cases allocate on an engine, since they write and read elements. Constant
fills and copies are compared by equality, since float32 holds every value used.

## AValueIsAllocatedOnceAndKeepsItsTensor

`materialise` of a [4, 6] value twice returns one tensor, which `tensor` returns too. The pool
holds one tensor, and the source received one request.

## TheBytesOfBoundValuesAreAccounted

After `materialise` of a [4, 6] value the pool books 96 bytes under `values`, its allocated and
unpacked bytes are equal, and it holds no plan.

## OnesAreSharedPerShapeAndType

`ones` for a float32 [3] returns the same tensor twice, and a float64 [3] returns another. The
pool books the bytes of both under `ones`.

## TheFlushFillsTheOnesTensorsWithOnes

After `flush_constants` every element of the float32 and the float64 ones tensor is 1.

## ScratchIsKeyedByNodeAndSlot

`scratch` for node 4, slot 0 returns the same tensor twice. Slot 1 of node 4 and slot 0 of node 5
return other tensors. The pool holds three tensors.

## ScratchBytesAreAccountedPerBuffer

Three float32 [2, 2] scratch buffers, outside a plan, book 48 bytes under `scratch`.

## AnAliasSharesTheStorageOfItsSource

A tensor from `create` is bound to one value and a second value aliases it, twice, which is
accepted. Values written through the first are read back through the second, and both have one
storage.

## AQueuedConstantIsWrittenByTheFlush

After `request_constant` of 2.5 on a created [4] tensor, the flush writes 2.5 to every element.

## AnEmptyFlushSucceeds

A flush with nothing queued succeeds.

## APoolOverASourceRefusesAMemoryPlan

`adopt` on a pool built over a `TensorSource` fails with `Unsupported` naming `TensorSource`, and
the pool holds no plan.

## APoolOverASourceRefusesAFlushOfQueuedConstants

After `ones` queued a fill, `flush_constants` on a pool built over a `TensorSource` fails with
`Unsupported`, naming `TensorSource` and the one queued fill.

## APoolOverASourceHasNoEngine

`engine()` on a pool built over a `TensorSource` stops the process with a report that names the
source and the missing engine. It is a death test.
