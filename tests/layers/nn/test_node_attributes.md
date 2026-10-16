# tests/layers/nn/test_node_attributes.cpp


Suites: `Small_NodeAttributes`.

Tests the readers in `NN::Detail` that a node factory reads a model-file row with.

This file needs no device; every case parses a row with `Config::parse_json` and reads it back.
`read_row_above` is pinned in `test_built_rows.cpp` and is not repeated here. Every refusal is
InvalidDescription, and each case checks that the message names the key, so a row author can
find the field.

## row

`row` parses a JSON text as the attributes of one row. `K_CONTEXT` names the row in every
diagnostic.

## ListNamesJoinsTheNamesWithCommas

Three names join as "filters, size, stride", and no names join as the empty string.

## ReadFeatureMapTakesOneRankThreeShape

One [4, 4, 3] input comes back unchanged. Two inputs are refused with "reads one input" and
"given 2"; a [12] input is refused naming the shape and "rank-3".

## RejectUnknownKeysNamesTheKeyAndTheAcceptedOnes

A row with only `filters` passes. A row that adds `pad` is refused naming the context, the key
and the accepted keys joined by `list_names`.

## ReadPositiveIntTakesOnlyAWholeNumberAboveZero

8 is read. An absent key is refused as "absent", 1.5 as not a "whole number" and 0 as not
"positive".

## TheDefaultingReadersFallBackOnAnAbsentKeyOnly

`read_positive_int_or` returns its fallback for an absent key and still refuses a stated -2.
`read_bool_or` returns its fallback for an absent key, the stated false over a true fallback, and
refuses 1 as not "true or false".

## ReadRealTakesAnyNumber

A whole 2 reads as 2.0 and 0.25 as 0.25, both exact. An absent key and a string are refused.

## TheNormReadersDefaultToTheLayersValues

An absent epsilon reads as `K_LAYER_NORM_EPSILON`, and a stated 0 is refused as not positive.
An empty row reads as a default `NormOptions`; a stated epsilon and momentum are read exactly; a
momentum of 1.5 is refused naming "[0, 1]".

## WrapBlockPresentsTheBlockAsAOnePortNode

`wrap_block` over a 6-to-8 `Linear` and a [6] input returns a node with one [8] port that reads
one value. Over a [5] input the `Linear` refuses the shape and `wrap_block` returns a failure
instead of a node.
