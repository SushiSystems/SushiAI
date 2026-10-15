# modules/vision/imaging/include/SushiAI/vision/imaging/affine.hpp


## AffineSettings, validate_affine

`scale` 0.5 and `translate` 0.1 (`cfg/default.yaml:119-120`, `data/augment.py:1102-1103`).
Rotation, shear and perspective are 0.0 by default (`cfg/default.yaml:118`, `121`, `122`) and are
not written, so the plan holds a scale and a translation and no general matrix builder. The
border value 114 is the reference's `borderValue` (`data/augment.py:1212`). `filter` and
`polygon_filter` are the two candidate filters of `boxes.hpp`.

## plan_affine

The reference builds `M = T @ S @ R @ P @ C` in `float32` (`data/augment.py:1144-1174`). With
`P`, `S` identity and angle 0 that is

```
scale       = uniform(1 - settings.scale, 1 + settings.scale)
translate_x = uniform(0.5 - settings.translate, 0.5 + settings.translate) * output.width
translate_y = uniform(0.5 - settings.translate, 0.5 + settings.translate) * output.height
m = [ scale, 0,     scale * (-source.width  / 2.0f) + translate_x,
      0,     scale, scale * (-source.height / 2.0f) + translate_y ]
```

`source` is the mosaic canvas, 2s by 2s, or the loaded image when the sample is not a mosaic.
`output` is s by s in both cases (`data/augment.py:2877`), which is why the training pipeline
needs no letterbox.

The reference makes eight draws here, five of them over a zero range (perspective twice, angle,
shear twice). `plan_affine` makes three: scale, translate x, translate y.

## apply_affine, image

The reference calls `cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, ...))` with no
`flags` (`data/augment.py:1212`). OpenCV documents the default as bilinear with a constant
border and an inverted `M`. This module's kernel, for output pixel (x, y):

```
sx = (x - m[2]) / m[0]     sy = (y - m[5]) / m[4]              (double)
x0 = floor(sx)   fx = sx - x0        y0 = floor(sy)   fy = sy - y0
wx1 = floor(fx * 2048 + 0.5)   wx0 = 2048 - wx1      (wy0, wy1 likewise)
out = (wy0 * (wx0 * p[y0][x0] + wx1 * p[y0][x0 + 1])
     + wy1 * (wx0 * p[y0 + 1][x0] + wx1 * p[y0 + 1][x0 + 1]) + 2^21) >> 22
```

A neighbour outside the source reads as `border_value`. Nothing is clamped, so the image fades
into the border over one pixel, and an output pixel with all four neighbours outside is exactly
`border_value`. There is no half-pixel offset: the warp maps integer coordinates, as the label
arithmetic does.

## apply_affine, labels

`data/augment.py:1231-1258`. In `float`, `x' = m[0] * x + m[2]` and `y' = m[4] * y + m[5]`.

- An object without a polygon: with no rotation the four warped corners of a box are its two
  warped corners, so the function maps `x1`, `y1`, `x2`, `y2`, clips with `clip_box` and tests
  with `plan.filter` (0.10).
- An object with a polygon (`apply_segments`, `data/augment.py:1295-1323`): every vertex is
  warped, the box is `visible_polygon_box` of the warped polygon clipped with `clip_box` to the
  output (`utils/instance.py:426-429`), the vertices are clamped to that box, and the test uses
  `plan.polygon_filter` (0.01). A polygon that shows
  nothing gives a zero box, which fails the filter.

Both tests compare `scale_box(source box, scale, scale)` with the new box. The source box of an
object with a polygon is the box its label carried into the warp, not one rebuilt from the
polygon, as in the reference.
