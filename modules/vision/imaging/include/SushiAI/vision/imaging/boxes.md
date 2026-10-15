# modules/vision/imaging/include/SushiAI/vision/imaging/boxes.hpp


## to_pixel_box, to_normalized_box

`to_pixel_box` is the reference's `xywh2xyxy` followed by `denormalize`
(`utils/ops.py:279-284`, `utils/instance.py:282`): corners first, then the multiply by the image
size. `to_normalized_box` is `xyxy2xywh` followed by the division in `Format`
(`utils/ops.py:257-260`, `data/augment.py:2487-2488`).

## clip_box, box_area

`clip_box` clamps x to [0, width] and y to [0, height] (`utils/instance.py:426-427`).
`box_area` is `(x2 - x1) * (y2 - y1)` (`utils/instance.py:99`).

## CandidateFilter, make_polygon_filter, is_box_candidate

The reference's `box_candidates` (`data/augment.py:1448-1451`), with its defaults
(`data/augment.py:1413-1416`): both sides above 2 pixels, aspect ratio below 100, every
comparison strict, `1e-16` added to each denominator. The area ratio the box must keep is 0.10
for an object without a polygon and 0.01 for an object with one (`data/augment.py:1255`); the
default `CandidateFilter` holds the first and `make_polygon_filter()` the second.

The reference picks 0.01 for every object of an image as soon as the image has any polygon.
Here the choice is per object. The two agree whenever a label file is all boxes or all
polygons, which is the only kind of file the parser accepts; they can differ only in a mosaic
that joins a box file with a polygon file, a mix the reference does not handle (its box and
polygon arrays then have different lengths).
