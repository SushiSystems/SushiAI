# tests/core/model_text/test_model_text.cpp


Suites: `Small_ModelText`.

Tests `attributes_from` and `model_from`, the parsers that turn the JSON text of a test into
config values.

This file needs no device.

## AttributesFromParsesAnObject

The text `{"units": 4, "bias": true}` parses to an object whose "units" is 4 and which has
"bias".

## AParseFailureNamesTheTestOrigin

Text cut off after a key throws an `Error` whose message names the origin "'<test>'" and says
the text is not valid JSON.

## ModelFromParsesAModel

A version 2 model with a [6] input, a "linear" row and a "gelu" row parses to an input of 6
elements and those two layers in order.
