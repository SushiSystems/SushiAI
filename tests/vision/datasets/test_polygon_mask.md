# tests/vision/datasets/test_polygon_mask.cpp


Suites: `Unit_DatasetsPolygonMask`.

Tests `paint_polygon_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/polygon_mask.hpp` over masks of 4 by
4 pixels. Each object is a rectangle given as a polygon and as its box.

## APolygonPaintsThePixelsWhoseCentresItHolds

A rectangle of class 2 from (1, 1) to (3, 3) over the background `K_MASK_VOID` gives a mask of
one channel whose four middle pixels are 2 and whose other twelve are 255.

## TheLaterOfTwoObjectsWinsAnOverlap

A rectangle of class 0 from (0, 0) to (3, 2), then one of class 1 from (2, 0) to (4, 4), over
the background 7 give the rows 0 0 1 1, 0 0 1 1, 7 7 1 1, 7 7 1 1.

## NoObjectIsTheBackgroundEverywhere

An empty label set over the background 3 gives sixteen bytes of 3.

## RefusesAnObjectItCannotPaint

An object with a box and no polygon fails with `InvalidDescription`, "paint_polygon_mask",
"object 0" and "no polygon". An object of class 255 fails with `InvalidDescription`,
"the class 255" and "0 to 254".
