# tests/core/data/test_batch_allocator.cpp


Suites: `Small_InMemoryBatchAllocator`, `Medium_EngineBatchAllocator`.

Tests `Data::InMemoryBatchAllocator` and `Data::EngineBatchAllocator`, the two adapters of the
`Data::BatchAllocator` port in `modules/core/data/include/SushiAI/data/batch_allocator.hpp`. The
`Small_` cases need no device. The `Medium_` fixture is an `EngineTest`. The layout of a whole
batch over either allocator is tested in `test_target.cpp`.

## ATensorHoldsTheExtentsAndTheTypeItWasAskedFor

The in-memory allocator returns a tensor of rank 2 with the extents 3 and 4 and the type int32
it was asked for.

## ATensorHoldsNoStorage

A tensor the in-memory allocator returns has no storage, so a case reads its layout and cannot
read an element.

## AnExtentThatIsNotPositiveIsRefused

The extents 3 and 0 fail with `InvalidDescription` and "positive".

## ATensorHoldsTheExtentsTheInMemoryAllocatorDescribes

The engine allocator returns a tensor with storage, and with the rank, extents and type the
in-memory allocator describes for the same request.
