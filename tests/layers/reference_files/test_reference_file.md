# tests/layers/reference_files/test_reference_file.cpp


Suites: `Small_ReferenceFile`, `Medium_ReferenceFileDeathTest`.

Tests `modules/layers/reference_files/include/SushiAI/reference_files/reference_file.hpp`: the
absent-file path of `ReferenceFile`, the element access of `ReferenceTensor`, and the comparisons
`near_reference` and `equal_reference`. The reading of a present file is exercised by the YOLO26
cases that open the exported references. No case reads a file: the absent file is reported through an
opener that finds nothing.

## float_tensor

Returns a found float32 tensor whose payload holds the given values.

## view

Views a vector of floats as a span.

## K_TOLERANCE_UNDER_TEST

The tolerance `near_reference` is called with. It is an input of the function under test, not a
claim about an accuracy.

## K_EXACT

The tolerance that demands equal values.

## AnAbsentFileIsReportedWithItsPath

A file the opener does not find, below the directory "fixtures", is not present, its reason names
"missing" and "fixtures/no_such_reference.sacp", and a read returns a tensor that is not found and
carries the same reason.

## ATensorIsReadAsItsOwnType

A float32 tensor yields its three values exactly.

## ATensorReadAsAnotherTypeStopsTheProcess

Reading a float32 tensor as int32 stops the process naming "float32" and "int32".

## ATensorThatWasNotFoundStopsTheProcessWhenRead

Reading a tensor that was not found stops the process naming "not found" and its reason.

## NearReferenceNamesTheFirstAndTheWorstMiss

Values within the tolerance pass, and equal values pass at the exact tolerance. Two misses out of four fail
with a message naming "2 of 4" and "worst at 3". Arrays of different lengths fail.

## EqualReferenceComparesAcrossIntegerTypes

int32 values equal to int64 ones pass. A difference at index 2 fails naming "first at 2", and
arrays of different lengths fail.
