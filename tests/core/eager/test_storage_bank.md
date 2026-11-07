# tests/core/eager/test_storage_bank.cpp


Suites: `Medium_StorageBank`, `Medium_StorageBankFatal`.

Tests that `StorageBank` holds, allocates and lets go of tensors by storage id.

This file needs an engine, because the bank allocates tensors.

## AllocateHoldsATensorOfTheShapeAndTypeAsked

The tensor has 12 elements of FLOAT64.

## AllocateCountsOneTensorAndOneAllocation

After one `allocate`, `size()` and `allocation_count()` are 1. The bookkeeping would run small
once the allocator is a port.

## AdoptSharesTheStorageOfTheCallersTensor

The bank's tensor has the address of the caller's and reads the caller's numbers. `adopt` does
not count as an allocation.

## ReleaseForgetsTheTensor

After `release` the bank holds no tensor under the id and the allocation count stays 1.

## ReleaseOfAnUnknownIdIsIgnored

`release` of an id the bank does not hold does nothing, which lets a session release the ids a
failed step never allocated.

## ASecondTensorUnderOneIdStopsTheProcess, AskingForAnUnknownIdStopsTheProcess

Both are caller errors and are fatal.
