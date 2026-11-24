# tests/layers/io/test_exported_model_writer.cpp

Suites: `Small_ExportedModelWriter`.

Tests `IO::write_exported_model`, which puts the files of an `ExportedModel` into one folder. Every case writes under a scratch directory that the test removes.

## WritesEveryFile

Two files are written with `ExistingFiles::Refuse`. The returned paths come back in file order and each holds its bytes.

## AnExistingFileIsRefusedBeforeAnyIsWritten

One of two files exists already. The call fails with `Incompatible`, naming the file and saying it exists. The other file is not written and the old file keeps its bytes.

## ForceReplacesAnExistingFile

With `ExistingFiles::Replace` the old file holds the new bytes.

## TwoFilesOfOneNameAreRefused

Two files of one name fail with `InvalidDescription`, naming the file twice. A name with a folder in it fails as no plain file name. Neither case writes a file.
