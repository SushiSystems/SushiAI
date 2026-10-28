# tests/vision/imaging/test_labels.cpp


Suites: `Small_ImagingLabels`.

Tests the parser of a label file's text.

This file needs no device and reads no file. `Small_ImagingLabels` passes strings to
`parse_label_text` of `modules/vision/imaging/include/SushiAI/vision/imaging/labels.hpp`. A row is a class and four
normalized box values, or a class and the x, y pairs of a polygon. The refusal cases check that
the message names the line and quotes the field.

## K_CLASS_COUNT and K_FRACTION_TOLERANCE

`K_CLASS_COUNT` is 80, the class count the cases parse with. `K_FRACTION_TOLERANCE` is 1e-7,
the tolerance of a parsed value.

## expect_box

`expect_box` compares the four values of a `NormalizedBox` within `K_FRACTION_TOLERANCE`.

## ParsesBoxRowsInFileOrder

Two rows give two objects, with classes 0 and 3 and the four values of each row.

## EmptyTextIsABackgroundImage

An empty string gives an empty set.

## TextOfBlankLinesIsABackgroundImage

Text of newlines, spaces, a tab and carriage returns gives an empty set.

## AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows

Rows with leading spaces, tabs, repeated spaces, a blank line between them and no final newline
parse to two objects.

## AcceptsCarriageReturnLineEndings

Rows ended by "\r\n" and by a lone "\r" parse to three objects.

## AcceptsVerticalTabAndFormFeedLineEndings

A vertical tab and a form feed each end a row.

## AcceptsTheSeparatorBytesAsLineEndings

The bytes 0x1c, 0x1d and 0x1e each end a row: four rows separated by them parse to four
objects.

## AcceptsTheUnitSeparatorByteBetweenFields

The byte 0x1f separates fields, alone or repeated.

## AcceptsExponentsBareDecimalPointsAndAPlusSign

"+7.0 5e-1 .5 2.5E-1 +0.125" parses to class 7 and the box (0.5, 0.5, 0.25, 0.125).

## RejectsARowOfFourColumns

A row of four columns returns `InvalidFile` with "parse_label_text: line 1", "5 columns" and "this row has 4".

## RejectsARowOfSixColumns

A row of six columns returns `InvalidFile` with "5 columns" and "this row has 6".

## RejectsARowOfOneColumn

A row of one column returns `InvalidFile` with "this row has 1".

## NamesTheLineOfTheRejectedRow

A short row after one row and one blank line is reported as "line 3". Blank lines count.

## RejectsAFieldThatIsNotANumber

The field "abc" returns `InvalidFile` with "'abc'" and "not a number".

## RejectsAFieldWithCharactersAfterTheNumber

The field "0.5x" returns `InvalidFile` with "'0.5x'".

## RejectsADecimalComma

The field "0,5" returns `InvalidFile` with "'0,5'".

## RejectsADoubledSign

The field "+-0.5" returns `InvalidFile` with "'+-0.5'".

## RejectsNotANumberAndInfinity

The fields "nan", "inf" and "1e999" each fail with the field quoted.

## RejectsAFiniteValueAboveTheFloatRange

The field "1e39", finite as a double and not as a float, returns `InvalidFile` with "not a finite number".

## RejectsANonzeroValueBelowTheSmallestNormalDouble

The fields "1e-310" and "1e-999" fail with the field quoted.

## AcceptsZeroAndTheSmallestNormalDouble

"0.0" and "2.2250738585072014e-308" are accepted and both store 0 within the tolerance.

## RejectsACoordinateBelowTheTolerance

Values of -0.02 and -0.5 fail with "negative".

## RejectsACoordinateAboveTheTolerance

Values of 1.02 and 640 fail with "above". The second is a pixel value in a file that must hold
fractions.

## StoresACoordinateInsideTheToleranceUnclipped

Values of 1.005 and -0.005 are stored as they are.

## AcceptsACoordinateOnEitherToleranceBound

Values of 1.01 and -0.01 are accepted and stored as they are.

## RejectsANegativeClass

Class -1 returns `InvalidFile` with "class" and "negative".

## RejectsAClassWithAFraction

Class 1.5 returns `InvalidFile` with "class" and "whole number".

## RejectsAClassAtOrAboveTheClassCount

Class 80 of 80 returns `InvalidFile` with "out of range" and "80 classes", and class 1 of 1 with "1 classes".

## AcceptsTheLastClass

Class 79 of 80 is accepted.

## RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder

Five rows of which two repeat earlier ones give three objects, with classes 5, 2, 9 in file
order.

## RemovesARowThatRepeatsAnotherInADifferentSpelling

"5.0 5e-1 0.50 .25 0.250" repeats "5 0.5 0.5 0.25 0.25" and is removed. Rows are compared by
value and not by text.

## KeepsRowsThatDifferInOneValue

Three rows that differ in the class or in one box value are all kept.

## ReducesAPolygonRowToTheBoxAroundItsPoints

A row of three points gives the box (0.5, 0.5, 0.5, 0.5) around them.

## KeepsThePointsOfAPolygonRowInFileOrder

A row of five points gives a polygon of those five points in file order.

## GivesABoxRowNoPolygon

Each object of two box rows has an empty polygon.

## ReducesAPolygonOfManyPointsToItsBounds

The five points of the earlier case give the box (0.5, 0.375, 0.75, 0.5).

## RejectsABoxRowInAPolygonFile

A box row after a polygon row returns `InvalidFile` with "line 2" and "polygon".

## RejectsAPolygonRowWithAHalfPoint

A polygon row of eight columns returns `InvalidFile` with "polygon" and "this row has 8".

## RejectsAPolygonRowOfFewerThanThreePoints

A row of three columns in a polygon file returns `InvalidFile` with "line 2", "three points" and "this row
has 3".

## RejectsAPolygonWhoseBoxLeavesTheImage

A polygon with a vertex at x = 1.5 returns `InvalidFile` with "above".

## RemovesARepeatedPolygonAndKeepsOneWithTheSameBox

Of three polygon rows the second repeats the first and is removed. The third has other points
and the same box, and is kept: both remaining boxes are (0.5, 0.5, 0.5, 0.5) and their second
vertices differ.
