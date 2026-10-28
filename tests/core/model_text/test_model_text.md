# tests/core/model_text/test_model_text.cpp


Suites: `Small_ModelText`.

Tests `attributes_from`, `model_from` and their `try_` forms, the parsers that turn the JSON text of a test into
config values.

This file needs no device.

## AttributesFromParsesAnObject

The text `{"units": 4, "bias": true}` parses to an object whose "units" is 4 and which has
"bias".

## AParseFailureNamesTheTestOrigin

Text cut off after a key makes `try_attributes_from` return `InvalidFile`, with a message that
names the origin "'<test>'" and says the text is not valid JSON.

## AFixtureThatIsNotJSONStopsTheProcess

`attributes_from` over that text and `model_from` over a document with no layers and no version
stop the process, with a report that names "'<test>'".

## TryModelFromReturnsTheRefusalOfAModelFile

`try_model_from` over a document with no version key returns `InvalidFile` naming "'<test>'" and
"sushiai_model".

## ModelFromParsesAModel

A version 2 model with a [6] input, a "linear" row and a "gelu" row parses to an input of 6
elements and those two layers in order.
