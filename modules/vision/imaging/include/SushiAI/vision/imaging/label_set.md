# modules/vision/imaging/include/SushiAI/vision/imaging/label_set.hpp


## LabelSet, FileLabels, Labels

`LabelSet<Box>` owns three lists of equal length: classes, boxes, polygons. The constructors
throw when the lengths differ, `add` appends to all three and `keep_if` removes from all three,
so no transform checks the lengths. `box(i)` and `polygon(i)` give one entry for writing; no
member changes one list's length alone.

## resample_polygons, to_pixel_labels, to_normalized_labels

`resample_polygons` applies `resample_polygon` to each polygon of one image with the count
`resampled_point_count` of the longest. `to_pixel_labels` applies `to_pixel_box` and multiplies
each vertex by the width and the height (`utils/instance.py:283-284`). A pipeline's output is
`Labels`; the code that builds loss targets calls `to_normalized_labels` when it wants the
reference's batch form.

## clip_labels, remove_empty_boxes

`clip_labels` applies `clip_box` and clamps every vertex to the image
(`utils/instance.py:426-429`). `remove_empty_boxes` keeps an object when its box area is above
zero (`utils/instance.py:449`).
