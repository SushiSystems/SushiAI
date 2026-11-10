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
