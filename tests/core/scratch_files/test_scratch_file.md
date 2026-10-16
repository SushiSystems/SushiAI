# tests/core/scratch_files/test_scratch_file.cpp


Suites: `Medium_ScratchFile`.

Tests that `ScratchFile` names a file under the temporary directory and removes it on
construction and on destruction.

This file writes to the temporary directory, which makes it medium. Every case uses the name
"sushiai_scratch_file_test.txt".

## ThePathIsTheNameUnderTheTemporaryDirectory

`path()` equals the temporary directory joined with the name.

## ConstructionRemovesALeftoverFile

A file written under the name before construction is gone once the `ScratchFile` exists.

## DestructionRemovesTheFile

A file the case writes at `path()` is gone after the `ScratchFile` leaves its scope.

## AFileNeverWrittenIsDestroyedQuietly

Constructing and destroying a `ScratchFile` whose file was never written throws nothing.
