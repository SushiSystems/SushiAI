# tests/core/scratch_files/test_scratch_directory.cpp


Suites: `Medium_ScratchDirectory`.

Tests that `ScratchDirectory` names a directory under the temporary directory and removes the
directory with everything below it on construction and on destruction.

This file writes to the temporary directory, which makes it medium. Every case uses the name
"sushiai_scratch_directory_test".

## ThePathIsTheNameUnderTheTemporaryDirectory

`path()` equals the temporary directory joined with the name.

## ConstructionLeavesTheDirectoryAbsent

A new `ScratchDirectory` creates nothing; its path does not exist.

## ConstructionRemovesALeftoverTree

A directory under the name with a nested folder and a file in each is gone once the
`ScratchDirectory` exists.

## DestructionRemovesTheWholeTree

A directory the case fills with a nested folder and two files is gone after the
`ScratchDirectory` leaves its scope.

## ADirectoryNeverCreatedIsDestroyedQuietly

Constructing and destroying a `ScratchDirectory` whose directory was never created throws nothing.
