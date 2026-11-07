# tests/vision/imaging/test_polygon.cpp


Suites: `Small_ImagingPolygon` and `Medium_ImagingPolygon`.

Tests polygon resampling, mapping, clamping, the point test and the visible box.

This file needs no device. `Small_ImagingPolygon` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/polygon.hpp`. The expected vertices and boxes are the outputs of the
reference's `resample_segments` and `segment2box` on the same inputs, written as literals; a
comment in each case names the call.

## Tolerances and K_IMAGE

`K_POINT_TOLERANCE` is 32 times the float unit roundoff, for a resampled coordinate of magnitude up to 4 that is not a dyadic fraction. `K_BOX_TOLERANCE` is 2048 times the float unit roundoff in px, for a box edge up to 100 that is not a dyadic fraction. `K_IMAGE` is 80 rows by 100 columns.

## expect_polygon and expect_polygon_exactly

`expect_polygon` compares a polygon with a flat list of expected x, y values within `K_POINT_TOLERANCE`; `expect_polygon_exactly` compares them exactly, for dyadic values.

## expect_box

`expect_box` compares the four edges of a `PixelBox` exactly.

## make_square, make_triangle and make_hexagon

`make_square` returns the square from (0, 0) to (4, 4). `make_triangle` returns a triangle in
fractions of an image. `make_hexagon` returns six vertices with integer coordinates.

## PointCountIsAThousandOrTheLongestPlusOne

`resampled_point_count` is 1000 for a longest polygon of 0, 3 or
1000 vertices and 1002 for 1001.

## ResampleMergesTheVerticesWithEvenPositions

The square resampled to 8 and to 11 points keeps its four vertices and gains the evenly spaced
positions between them.

## ResampleToOneMoreThanTheClosedLengthRepeatsTheFirstVertex

The square resampled to 6 points gains one point, at the first vertex.

## ResampleInterpolatesAlongTheEdges

The triangle resampled to 9 points holds interpolated points such as (0.55, 0.275) on its first
edge.

## ResampleOfALongPolygonTakesEvenPositionsOnly

The hexagon, closed, has seven points. Resampled to 4 it gives every second vertex and the
first again; resampled to 7 it gives the six vertices and the first again.

## ResampleLeavesAPolygonOfTheCountAlone

The triangle resampled to 3 points is unchanged.

## ResampleLeavesAnEmptyPolygonAlone

An empty polygon resampled to 1000 points stays empty.

## ResampleToAThousandPointsKeepsTheOutline

The triangle resampled to 1000 points has 1000 points, starts and ends at x = 0.1, and has the
bounds (0.1, 0.2, 0.7, 0.9).

## ResampleRejectsACountOfZero

Suite `Medium_ImagingPolygon`. A count of 0 stops the process with "resample_polygon" and "point count of 0".

## MapScalesThenTranslatesEachAxis

`map_polygon` with scales 2 and 0.5 and offsets 10 and -1 maps the square to the corners
(10, -1), (18, -1), (18, 1), (10, 1).

## ClampMovesVerticesOntoTheBounds

`clamp_polygon` to the box (0, 0, 4, 5) moves each coordinate outside onto the nearest bound
and leaves the vertex inside alone.

## PointInsideOrOnTheOutlineIsInside

`is_point_inside` is true for the centre of the square, a point on an edge and two vertices.

## PointOutsideIsOutside

`is_point_inside` is false for four points outside the square, for a point in the notch of a
concave polygon and for any point against an empty polygon. Two other points of the concave
polygon are inside.

## VisibleBoxOfAPolygonInsideTheImageIsItsBounds

A triangle inside `K_IMAGE` has the visible box (10, 20, 50, 60).

## VisibleBoxOfAPolygonOutsideTheImageIsZero

A triangle above and left of the image, one to its right and an empty polygon each give four
zeros.

## VisibleBoxUsesTheCrossingsOfTheImageLines

A triangle crossing the top and left edges gives (0, 0, 60, 50), and one crossing the right
and bottom edges gives (90, 70, 100, 80).

## VisibleBoxIncludesTheImageCornersThePolygonContains

A triangle that contains the image's origin gives (0, 0, 36.666664, 33), and a square around
the whole image gives the image, (0, 0, 100, 80).
