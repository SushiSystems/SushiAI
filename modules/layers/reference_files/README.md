# reference_files

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The reader of the YOLO26 reference set: the `.sacp` files, which use the checkpoint format,
and `MANIFEST.json`. `ReferenceFile` opens one file and indexes its records by tensor name.
`SA_OPEN_REFERENCE_OR_SKIP` opens a file or skips the test naming the missing file, and
`SA_READ_REFERENCE_OR_FAIL` reads one named tensor or fails the test. `near_reference` compares
float results within a tolerance and `equal_reference` compares integer results element by
element. The folder comes from `SA_TEST_FIXTURE_DIR`, which the test program defines.

## Depends on

`io`, for the checkpoint format, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/reference_files/reference_file.hpp` | `K_YOLO26_FIXTURE_FOLDER`, `fixture_directory`, `ReferenceTensor`, `ReferenceFile`, `near_reference`, `equal_reference`, `SA_OPEN_REFERENCE_OR_SKIP`, `SA_READ_REFERENCE_OR_FAIL` |
