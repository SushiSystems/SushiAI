# modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp


## fit_long_side

The reference's training and validation load (`data/base.py:297-299`): with
`r = image_size / max(height, width)` in `double`, a ratio of exactly 1 leaves the size alone,
and otherwise each side is `min(ceil(side * r), image_size)`. A small image is enlarged; the
reference has no guard against that.

## resize_bilinear

The reference calls `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` for the load
(`data/base.py:300`) and for the letterbox (`data/augment.py:1710`, `1839`). It never selects an
area filter on the default path, so this module declares none.

For destination column `x`, with `scale = source_width / width` in `double`:

```
sx = (x + 0.5) * scale - 0.5
x0 = floor(sx)            fx = sx - x0
if x0 < 0:                  x0 = 0;                 fx = 0
if x0 >= source_width - 1:  x0 = source_width - 1;  fx = 0
x1 = min(x0 + 1, source_width - 1)
wx1 = floor(fx * 2048 + 0.5)     wx0 = 2048 - wx1
```

Rows follow the same rule and give `y0`, `y1`, `wy0`, `wy1`. Each channel is then

```
out = (wy0 * (wx0 * p[y0][x0] + wx1 * p[y0][x1])
     + wy1 * (wx0 * p[y1][x0] + wx1 * p[y1][x1]) + 2^21) >> 22
```

in unsigned 32-bit arithmetic; the largest sum is 2048 * 2048 * 255 + 2^21, below 2^32. The
weight rounding and the blend are `Detail::to_weight`, `Detail::blend_pair` and
`Detail::blend_rows` in `modules/vision/imaging/source/bilinear.hpp`; the warp uses the same three.

## resize_nearest

Each destination pixel is a copy of one source pixel, so the function never produces a value the
source does not hold, which is what an image of class indices needs. Destination column `x` reads
source column `floor((x + 0.5) * source.width / width)`, computed in 64-bit integers as
`(2x + 1) * source.width / (2 * width)` and capped at the last column; rows likewise. The
function has no reference: the label pipeline of the reference carries polygons, not masks. It
stops the process through `check_source` and through `Image::reset`, which is where a size
outside the bounds of an image is refused. `Datasets::fit_mask` computes the same index with its own
kernel, because it writes into a region of a letterboxed mask.

## resize_long_side

`resize_bilinear` to `fit_long_side`, or a copy when the size does not change.
