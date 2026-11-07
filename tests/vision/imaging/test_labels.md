# tests/vision/imaging/test_labels.cpp


Suites: `Small_ImagingLabels` and the parameterised `Small_ImagingLabels<Family>` suites named below.

Tests the parser of a label file's text.

This file needs no device and reads no file. `Small_ImagingLabels` passes strings to
`parse_label_text` of `modules/vision/imaging/include/SushiAI/vision/imaging/labels.hpp`. A row is a class and four
normalized box values, or a class and the x, y pairs of a polygon. The refusal cases check that
the message names the line and quotes the field.

## K_CLASS_COUNT and K_FRACTION_TOLERANCE

`K_CLASS_COUNT` is 80, the class count the cases parse with. `K_FRACTION_TOLERANCE` is 2 times the float unit roundoff,
one float ulp at 1, the absolute tolerance of a parsed value.

## expect_polygon_points, RefusalRow and expect_refused

`expect_polygon_points` compares a polygon with listed points within `K_FRACTION_TOLERANCE`. A `RefusalRow` holds a
text, a class count and up to three substrings; `expect_refused` checks that the text returns `InvalidFile` naming them.

## expect_box

`expect_box` compares the four values of a `NormalizedBox` within `K_FRACTION_TOLERANCE`.

## ParsesBoxRowsInFileOrder

Two rows give two objects, with classes 0 and 3 and the four values of each row.

## TextWithNoRowIsABackgroundImage

Suite `Small_ImagingLabelsBackground`. An empty string and text of newlines, spaces, a tab and carriage returns each give an empty set.

## AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows

Rows with leading spaces, tabs, repeated spaces, a blank line between them and no final newline
parse to two objects.

## RowsAreSplitByEveryLineEnding

Suite `Small_ImagingLabelsLineEndings`. Rows ended by "\r\n" and by a lone "\r" parse to three objects; a vertical tab and a form feed each end a row; the bytes 0x1c, 0x1d and 0x1e each end a row, so four rows parse to four objects.

## AcceptsTheUnitSeparatorByteBetweenFields

The byte 0x1f separates fields, alone or repeated.

## AcceptsExponentsBareDecimalPointsAndAPlusSign

"+7.0 5e-1 .5 2.5E-1 +0.125" parses to class 7 and the box (0.5, 0.5, 0.25, 0.125).

## RejectsABoxRowOfTheWrongColumnCount

Suite `Small_ImagingLabelsColumnCount`. Rows of four, six and one column return `InvalidFile` naming "5 columns" and the count the row has.

## NamesTheLineOfTheRejectedRow

A short row after one row and one blank line is reported as "line 3". Blank lines count.

## RejectsAFieldThatIsNotANumber

Suite `Small_ImagingLabelsNotANumber`. The fields "abc", "0.5x", "0,5", "+-0.5", "nan", "inf" and "1e999" each return `InvalidFile` with the field quoted.

## RejectsAFiniteValueAboveTheFloatRange

The field "1e39", finite as a double and not as a float, returns `InvalidFile` with "not a finite number".

## RejectsANonzeroValueBelowTheSmallestNormalDouble

The fields "1e-310" and "1e-999" fail with the field quoted.

## AcceptsZeroAndTheSmallestNormalDouble

"0.0" and "2.2250738585072014e-308" are accepted and both store 0 within the tolerance.

## RejectsACoordinateOutsideTheToleranceBand

Suite `Small_ImagingLabelsCoordinateBand`. Values of -0.02 and -0.5 fail with "negative"; values of 1.02 and 640 fail with "above". The 640 is a pixel value in a file that must hold fractions.

## StoresACoordinateInsideTheToleranceUnclipped

Values of 1.005 and -0.005 are stored as they are.

## AcceptsACoordinateOnEitherToleranceBound

Values of 1.01 and -0.01 are accepted and stored as they are.

## RejectsAClassOutsideTheClassCount

Suite `Small_ImagingLabelsClassRange`. Class -1 fails with "negative", class 1.5 with "whole number", class 80 of 80 with "out of range" and "80 classes", and class 1 of 1 with "1 classes".

## AcceptsTheLastClass

Class 79 of 80 is accepted.

## RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder

Suite `Small_ImagingLabelsDuplicates`. Five rows of which two repeat earlier ones give three objects with classes 5, 2, 9 in file order; "5.0 5e-1 0.50 .25 0.250" repeats "5 0.5 0.5 0.25 0.25" and is removed, since rows are compared by value and not by text.

## KeepsRowsThatDifferInOneValue

Three rows that differ in the class or in one box value are all kept.

## ReducesAPolygonRowToTheBoxAroundItsPoints

Suite `Small_ImagingLabelsPolygonBounds`. A row of three points gives the box (0.5, 0.5, 0.5, 0.5); a row of five points gives the box (0.5, 0.375, 0.75, 0.5).

## KeepsThePointsOfAPolygonRowInFileOrder

A row of five points gives a polygon of those five points in file order.

## GivesABoxRowNoPolygon

Each object of two box rows has an empty polygon.

## RejectsABoxRowInAPolygonFile

A box row after a polygon row returns `InvalidFile` with "line 2" and "polygon".

## RejectsAPolygonRowOfTheWrongPointCount

Suite `Small_ImagingLabelsPolygonPointCount`. A polygon row of eight columns fails with "polygon" and "this row has 8"; a row of three columns in a polygon file fails with "line 2", "three points" and "this row has 3".

## RejectsAPolygonWhoseBoxLeavesTheImage

A polygon with a vertex at x = 1.5 returns `InvalidFile` with "above".

## RemovesARepeatedPolygonAndKeepsOneWithTheSameBox

Of three polygon rows the second repeats the first and is removed. The third has other points
and the same box, and is kept: both remaining boxes are (0.5, 0.5, 0.5, 0.5) and their second
vertices differ.
