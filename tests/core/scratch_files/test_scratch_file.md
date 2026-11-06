# tests/core/scratch_files/test_scratch_file.cpp


Suites: `Small_ScratchFile`, `Medium_ScratchFile`.

Tests that `ScratchFile` names a file under the temporary directory and removes it on
construction and on destruction.

The case that composes a path touches nothing and is small. The others write to the temporary
directory, which makes them medium. Every case uses the name "sushiai_scratch_file_test.txt".

## ThePathUnderABaseIsTheNameBelowIt

`ScratchFile::path_under` gives the base joined with the name.

## ConstructionUnderABaseRemovesTheLeftoverFileOfThatBase

A `ScratchFile` made with a base and a name has that base and name as its path, and removes a
file a previous run left there. The base is a `ScratchDirectory`.

## ConstructionRemovesALeftoverFile

A file written under the name before construction is gone once the `ScratchFile` exists.

## DestructionRemovesTheFile

A file the case writes at `path()` is gone after the `ScratchFile` leaves its scope.
