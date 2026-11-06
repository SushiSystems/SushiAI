# tests/core/scratch_files/test_scratch_directory.cpp


Suites: `Small_ScratchDirectory`, `Medium_ScratchDirectory`.

Tests that `ScratchDirectory` names a directory under the temporary directory and removes the
directory with everything below it on construction and on destruction.

The case that composes a path touches nothing and is small. The others write to the temporary
directory, which makes them medium. Every case uses the name "sushiai_scratch_directory_test".

## ThePathUnderABaseIsTheNameBelowIt

`ScratchDirectory::path_under` gives the base joined with the name.

## ConstructionUnderABaseRemovesTheLeftoverTreeOfThatBase

A `ScratchDirectory` made with a base and a name has that base and name as its path, and
removes a tree a previous run left there.

## ConstructionLeavesTheDirectoryAbsent

A new `ScratchDirectory` creates nothing; its path does not exist.

## ConstructionRemovesALeftoverTree

A directory under the name with a nested folder and a file in each is gone once the
`ScratchDirectory` exists.

## DestructionRemovesTheWholeTree

A directory the case fills with a nested folder and two files is gone after the
`ScratchDirectory` leaves its scope.
