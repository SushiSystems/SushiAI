# tests/core/graph/test_cast_insertion.cpp


Suites: `Small_CastInsertion`.

Tests the parts of the cast pass that `tests/core/autograd/test_backward.cpp` does not: the
report of a pass that converts nothing, a narrow graph under the mixed policy, and the exact
text of `describe`. The conversions the pass inserts are pinned there. No device is used.

## APassWithNothingToConvertGivesTheShortReport

`insert_casts` with no options over a float32 RELU graph reports no cast, no conversion into
half, the same node count before and after, and the policy name "fp32". Its `to_string()` holds
"casts inserted : 0" and no "values retyped" line.

## AGraphAlreadyNarrowHasNoMismatchUnderTheMixedPolicy

A product of a half [4, 3] input and a half [5, 3] weight agrees with `MixedPrecisionPolicy`:
`find_dtype_mismatches` returns nothing.

## TheDescriptionNamesTheNodeTheTypesAndTheSide

A mismatch on result 7 of node 3, a MATMUL found float32 where half is expected, is described as
"node 3 (", the operation's name, ") computes in half but its result value 7 is float32", the
type names taken from `dtype_name`. With `is_result` false the text says "its operand value 7".

## APolicyAskingForAnIntegerConversionIsRefused

A policy that asks a float32 RELU to compute in int32 makes `insert_casts` return
`InvalidDescription` naming the operation, both type names and "real floating types". The
graph keeps its node and value counts.

## ANodeWithoutAResultIsFatal

A RELU node added by hand with no result stops the process in `insert_casts` with a report
naming the operation and "declares no result".
