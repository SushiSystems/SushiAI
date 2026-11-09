# tests/vision/importers/test_darknet_config.cpp


Suites: `Small_DarknetConfig`, `Small_DarknetConfigMalformedSection` and
`Medium_DarknetConfig`.

Tests `parse_darknet_config`, which reads a config text into a `DarknetConfig` of one network
section and a list of layer sections, each with its options and line numbers, and
`load_darknet_config`. Every refusal of the parser has the code `InvalidDescription`.

## SkipsCommentsAndBlankLinesAndCountsThem

A text with a `#` comment line, a `;` comment line, a blank line and a comment between two
keys parses to a "network" section on line 4 with "width" on line 5 and "height" on line 7.

## DropsTheWhitespaceAroundTheSignAndTheLine

Spaces around `=`, tabs around `=`, an indented section line and trailing spaces on a value
are dropped: "width" and "height" are "416" on lines 2 and 3, and the indented
"convolutional" layer on line 4 has "size" "3" on line 5.

## AListValueLosesItsInnerSpaces

"10, 14,  23 ,27" reads as "10,14,23,27", "-1 , 61" as "-1,61" and "0,1 , 2" as "0,1,2".

## AnEmptyValueStaysEmpty

A key with nothing after `=` has the value "" on its line.

## ASecondEqualsSignStaysInTheValue

"expr=a=b" has the value "a=b".

## IgnoresCarriageReturnsAtLineEnds

A text with CRLF line ends parses to the same keys and values, with "width" on line 2, the
layer on line 3 and "filters" on line 4.

## AcceptsOneKeyInTwoSections

"width" in the network section and again in a layer section parses, and the layer's value is
"8" on line 4.

## RefusesAKeyBeforeAnySection

A key on line 1 before any section fails with a message holding "test.cfg:1" and
"key 'width'".

## RefusesALineThatIsNeitherSectionNorOption

A line without `=` fails with a message holding "test.cfg:2".

## RefusesAnEmptyKey

A line that starts with `=` fails with a message holding "test.cfg:2".

## RefusesAKeyRepeatedWithinASection

"width" twice in one section fails with a message holding "test.cfg:3", "key 'width'" and
"on line 2".

## RefusesAFirstSectionThatIsNotNetOrNetwork

A text whose first section is `[convolutional]` fails with a message holding "test.cfg:1" and
"[convolutional]".

## RefusesAMalformedSectionLine

One test per row. A section line without its closing bracket fails with a message holding
"test.cfg:1" and "']'". An empty section name and text after the closing bracket each fail on
"test.cfg:1".

## RefusesATextWithoutASection

A text of one comment line fails with a message holding "test.cfg" and "[net]".

## MissingFileGivesFileNotFound

`load_darknet_config` of a path that does not exist fails with `FileNotFound` and names the
path. The open of the file is the promise, so the test is `Medium_`.
