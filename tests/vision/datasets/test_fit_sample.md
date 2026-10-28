# tests/vision/datasets/test_fit_sample.cpp


Suites: `Small_DatasetsFitSample`.

Tests `fit_sample` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_sample.hpp`.
A box corner may differ from its expected value by 1e-4 pixels.

## ResizeScalesTheLabelsWithEachAxis

A gray image of 2 rows and 4 columns at level 100, resized to 4 by 4, gives 16 bytes of 100
in one channel. The fitted sample keeps the file index 6 and the class index 3. Its one
object keeps the class 1, and the box of centre (0.5, 0.5) and size (0.5, 0.5) becomes the
corners (1, 1) and (3, 3).

## LetterboxPadsAndMovesTheLabels

The same shape at level 200, letterboxed to 4 by 4, has rows 1 and 2 at 200 and rows 0 and 3
at 114. The sample has no class index, and the box of centre (0.5, 0.5), width 0.5 and height
1.0 becomes the corners (1, 1) and (3, 3).

## AnObjectWithoutAreaIsRemoved

Of two objects, the one of class 2 with a width of 0 is removed and the one of class 5 is
kept.

## AnInputSizeOutsideTheBoundsIsRefused

A target size of 0 by 4 fails with `InvalidDescription`, "fit_sample" and "0 by 4".

## ASourceOfNoPixelsStopsTheProcess

A default `ImageView` as the source stops the process with "resize_bilinear" and "no pixels".

## ALetterboxThatLeavesNoPixelsIsRefused

A source of 1 row and 4000 columns letterboxed to 4 by 4 would resize its rows to 0 pixels. The
fit returns `InvalidDescription` with "plan_letterbox" and "zero pixels", the status of the plan
passed on.
