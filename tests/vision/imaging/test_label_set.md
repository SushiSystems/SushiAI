# tests/vision/imaging/test_label_set.cpp


Suites: `Small_ImagingLabelSet`, `Medium_ImagingLabelSet`.

Tests the label set that holds classes, boxes and polygons in step, and its conversions.

This file needs no device. `Small_ImagingLabelSet` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/label_set.hpp`. `Labels` holds pixel boxes and `FileLabels` holds
normalized boxes; both keep one class, one box and one polygon per object, and a polygon may be
empty.

## Tolerance and K_SIZE

`K_FRACTION_TOLERANCE` is 32 float unit roundoffs, about 1.9e-6 of the image: a resampled vertex is
one interpolation of a few float operations on values below 1. Every other case has dyadic values
and compares exactly. `K_SIZE` is 480 rows by 640 columns.

## expect_pixel_box, expect_normalized_box and expect_point

The three helpers compare a `PixelBox`, a `NormalizedBox` and a `Point` with expected values,
exactly.

## make_diagonal

`make_diagonal` returns a polygon of a given number of vertices spread along the diagonal from
(0, 0) towards (1, 1).

## make_labels_with_empty_boxes

`make_labels_with_empty_boxes` returns six objects, classes 10 to 15: those of classes 11 and 13
have zero area, the one of class 15 has a negative area, and the other three have a positive area.
The objects of classes 11 and 12 carry polygons of one and two vertices.

## ConstructorsRejectListsOfDifferentLengths

Two classes with one box, one class with two boxes, two objects with one polygon, and no class
with one box all stop the process with a message containing "LabelSet" and the two counts. The test
is in `Medium_ImagingLabelSet` because a death test starts a child process.

## SetWithoutPolygonsHoldsOneEmptyPolygonPerObject

A set built from two classes and two boxes has two polygons, both empty.

## AddAppendsOneEntryToEachList

`add` with and without a polygon grows the classes, the boxes and the polygons together. The
object added without one has an empty polygon.

## KeepIfRemovesAnObjectFromAllThreeLists

`keep_if` with a predicate that keeps even classes leaves classes 2 and 4 with their own boxes
and polygons.

## ToPixelLabelsKeepsClassesAndOrder

`to_pixel_labels` converts three normalized boxes in `K_SIZE` and keeps the classes 7, 2, 7 in
order.

## ToPixelLabelsScalesVerticesByWidthAndHeight

A vertex at (0.25, 0.25) becomes (160, 120): x is multiplied by the width and y by the height.
An empty polygon stays empty.

## ToNormalizedLabelsKeepsClassesAndOrder

`to_normalized_labels` converts two pixel boxes and the vertices of one polygon, and keeps
classes and order.

## LabelConversionsOfABackgroundImageAreEmpty

Both conversions of an empty set are empty.

## ClipLabelsClipsBoxesAndVerticesAndRemovesNone

`clip_labels` clips each box to the image and clamps each vertex onto it. A box wholly outside
collapses to zero width and stays in the set.

## RemoveEmptyBoxesDropsZeroAndNegativeAreaBoxes

Of six objects, the two with zero width or zero height and the one whose x1 is above its x2 are
removed. The boxes (0, 0, 5, 5), (1, 1, 2, 3) and (7, 7, 9, 8) remain.

## RemoveEmptyBoxesKeepsTheClassesOfTheSurvivorsInOrder

The classes 10, 12 and 14 remain, in order.

## RemoveEmptyBoxesKeepsEachSurvivorsPolygon

The survivors keep their own polygons: none for class 10, two vertices for class 12 and none for
class 14.

## ResamplePolygonsGivesEachPolygonAThousandPoints

`resample_polygons` gives a triangle and a polygon of 40 vertices `K_POLYGON_POINTS` points
each. An empty polygon stays empty, and classes and boxes are unchanged.

## ResamplePolygonsLeavesASetWithoutPolygonsAlone

A set whose one polygon is empty comes back with the polygon empty and the box unchanged.
