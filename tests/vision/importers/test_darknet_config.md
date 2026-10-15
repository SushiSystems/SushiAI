# tests/vision/importers/test_darknet_config.cpp


Suites: `Unit_DarknetConfig`.

Tests `parse_darknet_config`, which reads a config text into a `DarknetConfig` of one network
section and a list of layer sections, each with its options and line numbers, and
`load_darknet_config`. Every refusal of the parser has the code `InvalidDescription`.

## ParsesEveryLexicalFormOfTheGrammar

A text with `#` and `;` comment lines, a blank line, indented lines, tabs and spaces around
`=`, and spaces inside comma lists parses to a "network" section on line 4 with four options
and to three layers, "convolutional", "route" and "yolo". Each option has its key, its value
and its line. A list value loses its inner spaces ("10,14,23,27"), an empty value stays empty,
and a value keeps a second `=` ("a=b").

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

A section line without its closing bracket fails with a message holding "test.cfg:1" and
"']'". An empty section name and text after the closing bracket each fail on "test.cfg:1".

## RefusesATextWithoutASection

A text of one comment line fails with a message holding "test.cfg" and "[net]".

## MissingFileGivesFileNotFound

`load_darknet_config` of a path that does not exist fails with `FileNotFound` and names the
path.
