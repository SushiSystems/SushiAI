# tests/core/core/test_allocation.cpp


Suites: `Medium_Allocation`.

Tests what `allocate_tensor`, `allocate_like` and `allocate_arena` give back and what they
refuse. Every case allocates on the engine `EngineTest` gives it, on the CPU SYCL device. The
views `view_tensor` carves from an arena are covered by `tests/core/graph/test_arena.cpp`.

## ATensorHasTheExtentsAndTypeItWasAskedFor

A float64 tensor of `Shape{2, 3, 4}` has rank 3, those extents, 24 elements and its own storage
of at least 24 times `dtype_size(DType::Float64)` bytes.

## AnExtentThatIsNotPositiveIsRefused

The extents 4 and 0, passed as a span, are refused with `StatusCode::InvalidDescription`.

## ALikeTensorCopiesTheDescriptionAndSharesNothing

`allocate_like` over an int32 tensor of `Shape{5, 2}` gives a tensor with the same rank,
extents, element type and layout, and a storage that is not the original's.

## AnArenaHoldsAtLeastTheBytesAskedFor

An arena of `K_ARENA_BYTES`, 1000 bytes, is a storage of at least that many bytes.
