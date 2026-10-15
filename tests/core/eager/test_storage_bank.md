# tests/core/eager/test_storage_bank.cpp


Suites: `Unit_StorageBank`.

Tests that `StorageBank` holds, allocates and lets go of tensors by storage id.

This file needs an engine, because the bank allocates tensors.

## AllocateHoldsATensorOfTheShapeAndTypeAsked

The tensor has 12 elements of FLOAT64, and `allocation_count()` is 1.

## AdoptSharesTheStorageOfTheCallersTensor

The bank's tensor has the address of the caller's and reads the caller's numbers. `adopt` does
not count as an allocation.

## ReleaseForgetsTheTensorAndIgnoresAnUnknownId

`release` of an id the bank does not hold does nothing, which lets a session release the ids a
failed step never allocated.

## ASecondTensorUnderOneIdStopsTheProcess, AskingForAnUnknownIdStopsTheProcess

Both are caller errors and are fatal.
