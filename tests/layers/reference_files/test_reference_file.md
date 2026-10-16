# tests/layers/reference_files/test_reference_file.cpp


Suites: `Small_ReferenceFile`.

Tests `modules/layers/reference_files/include/SushiAI/reference_files/reference_file.hpp`: the
absent-file path of `ReferenceFile`, the element access of `ReferenceTensor`, and the comparisons
`near_reference` and `equal_reference`. The reading of a present file is exercised by the YOLO26
cases that open the exported references. No case reads a file.

## float_tensor

Returns a found float32 tensor whose payload holds the given values.

## view

Views a vector of floats as a span.

## AnAbsentFileIsReportedWithItsPath

`fixture_directory()` equals `SA_TEST_FIXTURE_DIR`. A file that is not there is not present, its
reason names "missing" and the file, and a read returns a tensor that is not found and carries
the same reason.

## ATensorIsReadOnlyAsItsOwnType

A float32 tensor yields its three values exactly. Reading it as int32 stops the process naming "float32" and
"int32", and reading a tensor that was not found stops it naming "not found" and its reason.

## NearReferenceNamesTheFirstAndTheWorstMiss

Values within 0.1 pass, and equal values pass at a tolerance of 0. Two misses out of four fail
with a message naming "2 of 4" and "worst at 3". Arrays of different lengths fail.

## EqualReferenceComparesAcrossIntegerTypes

int32 values equal to int64 ones pass. A difference at index 2 fails naming "first at 2", and
arrays of different lengths fail.
