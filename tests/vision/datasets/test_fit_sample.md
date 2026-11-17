# tests/vision/datasets/test_fit_sample.cpp


Suites: `Small_DatasetsFitSample`, `Small_DatasetsFitSampleLetterboxRow` and
`Medium_DatasetsFitSampleFatal`.

Tests `fit_sample` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_sample.hpp`.
A box corner may differ from its expected value by 1e-4 pixels.

## ResizeScalesTheLabelsWithEachAxis

A gray image of 2 rows and 4 columns, resized to 4 by 4, keeps one object of class 1, and its
box of centre (0.5, 0.5) and size (0.5, 0.5) becomes the corners (1, 1) and (3, 3).

## ResizeFillsTheImageAndCarriesTheFileAndTheClass

The same image at level 100, resized to 4 by 4, gives 16 bytes of 100 in one channel. The
fitted sample keeps the file index 6 and the class index 3.

## LetterboxPadsTheShortAxis

`Small_DatasetsFitSampleLetterboxRow`, one row each. A gray image of 2 rows and 4 columns at
level 200, letterboxed to 4 by 4, has rows 0 and 3 at 114 and rows 1 and 2 at 200.

## LetterboxMovesTheLabelsByThePadding

The sample has no class index, and the box of centre (0.5, 0.5), width 0.5 and height 1.0
becomes the corners (1, 1) and (3, 3).

## AnObjectWithoutAreaIsRemoved

Of two objects, the one of class 2 with a width of 0 is removed and the one of class 5 is
kept.

## AnInputSizeOutsideTheBoundsIsRefused

A target size of 0 by 4 fails with `InvalidDescription`, "fit_sample" and "0 by 4".

## ASourceOfNoPixelsStopsTheProcess

Size (RW-166): the death check runs in a child process and took more than the 0.1 s of a
`small` test on the debug build, so the case is in `Medium_DatasetsFitSampleFatal`; a death test is `medium` by its
size rule.

A default `ImageView` as the source stops the process with "resize_bilinear" and "no pixels".

## ALetterboxThatLeavesNoPixelsIsRefused

A source of 1 row and 4000 columns letterboxed to 4 by 4 would resize its rows to 0 pixels. The
fit returns `InvalidDescription` with "plan_letterbox" and "zero pixels", the status of the plan
passed on.

## DarknetLetterboxMatchesTheHandDerivedImage

A gray image of 2 rows and 3 columns at the levels 0, 30, 60 and 90, 120, 150 (30 c + 90 r),
fitted to 6 by 6 with `SampleFit::DarknetLetterbox`. The width ratio 6 / 3 = 2 is below the
height ratio 6 / 2 = 3, so the image becomes 4 rows of 6 columns with one row of 128 above and one
below. The kernel reads columns at 0.4 k and rows at r / 3, so row r holds 12 k + 30 r:
0, 12, 24, 36, 48, 60 for r = 0, and 30 more for each row down. The box of centre (0.5, 0.5) and
size (0.5, 0.5), the corners (0.75, 0.5) and (2.25, 1.5), becomes (1.5, 2) and (4.5, 4). The fit
record has the source 2 by 3, the scales 2 and 2 and the offsets 0 and 1.

## DarknetLetterboxTruncatesTheSidesAndTheOffsets

A gray image of 3 rows and 5 columns at the levels 10 c + 100 r, fitted to 4 by 6. The width
ratio 1.2 is below 1.333, so the width is 6 and the height is the truncated 3 * 6 / 5 = 3, with
the one padding row below and none above: (4 - 3) / 2 truncates to 0. The columns are read at
0.8 k, so rows 0 to 2 are 8 k + 100 r and row 3 is 128; the fit record has the offset 0 on the
rows.
