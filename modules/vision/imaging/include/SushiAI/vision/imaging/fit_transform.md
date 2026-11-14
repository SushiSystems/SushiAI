# modules/vision/imaging/include/SushiAI/vision/imaging/fit_transform.hpp

## FitTransform

A fit puts a source image in the model's input by a scale and an offset per axis, so a point of the
source at (x, y) lies at (x * scale_x + offset_x, y * scale_y + offset_y) in the fitted image. A
resize has two scales and no offset. A letterbox has one scale, the plan's ratio, and the padding
above and left of the image as the offsets; `apply_letterbox` of a `Labels` computes exactly this
for each box, in `float`.

## box_in_source

Inverts the map above for the corners of one box and clips each corner to [0, source width] and
[0, source height]. A box that lies wholly in the padding collapses to a line on the edge of the
source and has no area; the caller decides whether to print it. The map is exact when the forward
map was, for example on a ratio and offsets that are dyadic, and otherwise within one rounding of
`float` per corner.
