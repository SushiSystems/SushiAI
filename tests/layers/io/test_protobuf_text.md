# tests/layers/io/test_protobuf_text.cpp

Suites: `Small_ProtobufText`, `Small_ProtobufTextScalar`, `Small_ProtobufTextString`,
`Small_ProtobufTextRefusal`.

Tests `IO::parse_protobuf_text`, in `modules/layers/io/include/SushiAI/io/protobuf_text.hpp`, on
documents written in the test. A tree is compared as the text `dump` writes: `name=value` for a
scalar, a string in double quotes, `name{...}` for a message, fields joined by `;`. Each expected
text is worked out by hand from the document. Columns count bytes from 1.

## ParsesALeNetPrototxtToItsTree

A LeNet-style document of five layers (input, convolution with two `param` blocks and two fillers,
pooling with the enum `MAX`, ReLU, softmax with `include { phase: TEST }`) parses to the tree its
braces describe, with repeated `layer`, `param` and `dim` fields in document order.

## FindsRepeatedFieldsInDocumentOrder

`find_fields` returns the five `layer` fields in order; `find_field` of the second reaches
`convolution_param` and `kernel_size` 5; two `param` fields; a name no field holds gives null.

## RecordsWhereEachFieldStarts

In `a: 1` then `  bb { c: 2 }` the fields start at line 1 column 1, line 2 column 3, and `c` at
line 2 column 8.

## TakesTheKindItsFormGives

Rows over integers, signs, hexadecimal, decimals, exponents, the float suffix, a trailing point and
`-inf`, all numbers; the bare words `nan`, `MAX` and `true` are identifiers.

## DecodesToTheBytesItWrites

Rows over `\n`, both quotes, `\"`, `\\`, `\x41\102` (A and B), a one-digit hex escape, `é`
(C3 A9), `\U0001F600` (F0 9F 98 80), joined strings, strings joined across a comment, and the empty
string.

## IgnoresCommentsAndFieldSeparators

`#` comments, `;` and `,` between fields and an empty `c { }` give `a=1;b=2;c{};d=x`.

## ReadsAMessageWithAndWithoutAColon

`a: { b: 1 }` and `a { b: 1 }` both give `a{b=1}`.

## ParsesAnEmptyDocumentToAnEmptyRoot

An empty document and one of a comment alone give a root with no fields.

## ParsesMessagesNestedToTheLimit

One hundred nested `a{` parse; the chain of first fields is 100 deep.

## RefusesMessagesNestedPastTheLimit

One hundred and one nested `a{` fail at line 1 column 202, the 101st brace, naming the limit of 100.

## NamesTheLineAndColumnItFailsAt

Rows, each failing with `InvalidFile` naming the origin, the line and column and the cause: a stray
`}`, an unclosed brace of an outer and of an inner message (the column of the opening brace), a bad
token, a brace or a bracket where a value belongs, a missing colon, a digit where a name belongs, a
control byte, a malformed number, a lone sign, a value followed by junk, an unterminated string and
one cut by the end, an unknown escape, an octal escape above 255, a short `\u` escape, an escape cut
by the end, and a document that ends after a name and after a colon.
