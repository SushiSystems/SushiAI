# imaging

The host-side data pipeline of the YOLO26 programme: label parsing, the raw image file, resize,
letterbox, mosaic, random affine, HSV jitter and horizontal flip, composed into a training
pipeline and a validation pipeline. Everything works on `uint8` pixels and `float` boxes in host
memory. Nothing here reaches an engine, SYCL or a third-party library, so every unit is tested
without a device. The module depends on `core` for `Error` and `SA_THROW_IF` and on the C++
standard library. `affine_nearest.hpp` and `mask_pipeline.hpp` return a `Status` and include
`<SushiAI/core/status.hpp>`, so `core` has to become a public dependency;
`modules/vision/imaging/CMakeLists.txt` still declares it private.

Each header has one source under `modules/vision/imaging/source/` and one test file under
`modules/vision/imaging/tests/`, named `test_imaging_<header>.cpp`. Four files break that pairing:
`partner_source.hpp` is an interface with no source; `test_imaging_partners.cpp` tests both of
its implementations; `test_imaging_mask_pipeline.cpp` tests `affine_nearest.hpp` and
`mask_pipeline.hpp`, which have not run; `modules/vision/imaging/source/bilinear.hpp` and `bilinear.cpp` are private and hold the
fixed-point arithmetic the resize, the letterbox and the warp share. The unit files are compiled
into `sushiai_tests`, which passed all 1803 cases in the run of 2026-10-09
(`docs/agent/2026_10_08_SUSHIBLAS_MIGRATION/REPORT.md`).

The rulings this module implements are in `docs/design/PROGRAMME.md` and
`docs/agent/2026_10_07_YOLO26/SPEC.md`, section 8: the default augmentation pipeline in full and
none of the pieces that are off by default; statistical equivalence with the reference, not bit
parity with OpenCV; images decoded once into a raw cache by a separate tool.

"The reference" below is the Ultralytics package at commit `ce13994`. A citation such as
`data/augment.py:1255` is a path and a line inside that package.

## Module facts

**Layout.** An image is height by width by channels, row-major, channels innermost (HWC).

**Channel order and count.** A three-channel image is RGB, red at channel 0, in every function
of the module. The reference works in BGR and reverses the channels in its `Format` step
(`data/augment.py:2520`). This module never holds BGR: `read_raw_image_pixels` swaps a BGR file
as it reads it. Every sample a pipeline produces has three channels. The reference loads with
`IMREAD_COLOR` unless the dataset is declared one-channel (`data/base.py:153`), so a grey file
reaches its transforms as three equal channels; here a source may hand over one channel and the
pipeline expands it with `expand_gray` at load. A source image with two or four channels is
refused.

**Where float begins.** Outside the module. A pipeline returns `uint8` RGB in HWC, which is
already the layout of the model's NHWC input. The caller that assembles a batch stacks the
images and divides by 255, with no mean and no standard deviation, as the reference does at
`models/yolo/detect/train.py:119`.

**Coordinates.** Pixel (x, y) is the point (x, y), and an image of width W spans [0, W]. The
warp, the boxes and the polygons use the same convention, so a box edge at x maps where pixel x
maps. The resize alone uses pixel centres at x + 0.5.

**Labels.** One object is a class, a box and a polygon; `LabelSet` holds the three lists and
keeps them the same length. `NormalizedBox` is the file form, centre and extent as fractions of
the image. `PixelBox` is two corners in pixels. Every transform takes and returns `Labels`, the
pixel form, in the coordinates of its own input and output image. The reference carries
normalised boxes into its first transform and converts there (`data/augment.py:722-723`,
`1231-1232`, `1929-1930`); here the pipeline calls `to_pixel_labels` once, right after the load,
with the loaded size. The arithmetic is the same and in the same order.

**Arithmetic.** Boxes and polygons are `float`, as the reference's are `float32`. Pixel kernels
are integer arithmetic with positions computed in `double`. The library is built with
`-ffp-contract=off`, so no product and sum is fused, and a seed gives the same sample on every
platform.

**Transform shape.** Each transform has a settings struct the caller fills, a `validate_`
function that throws on a setting out of range, a plan struct that holds everything the
transform will do, a `plan_` function that calls `validate_`, then makes every random draw, and
`apply_` functions that are pure: one for the image, one for the labels. A test checks the
draws by reading the plan, and replays a transform by passing the same plan twice. A pipeline
calls every `validate_` in its constructor, so a bad setting fails on the caller's thread before
any image is loaded.

Three transforms differ from that shape, each for a stated reason. `plan_letterbox` takes no
generator because a letterbox draws nothing, and `validate_letterbox` accepts everything because
neither of its two fields has an invalid value. `apply_hsv` has no label function because a
colour change moves no label. Mosaic has two steps before `apply_`, `draw_mosaic` and
`plan_mosaic`, because the partner indices must be drawn before the partner images can be
loaded and the placement needs their sizes.

**Sources and destinations.** An image `apply_` reads an `ImageView` and writes an `Image` the
caller owns. It calls `check_source` before it touches the destination, and that function is
the one aliasing rule of the module: the source has pixels, a valid size and a valid channel
count, and reads no byte of the destination's allocation. A view of some rows of the
destination is refused like the destination itself. The whole allocation counts, not the
current size, because an `Image` keeps its bytes when it is reset to a smaller size. An image
`apply_` allocates nothing once its destination has grown to size, so a pipeline reuses its
buffers for a whole epoch; the two exceptions are the tap tables of the resize and of the warp,
one small vector per axis and per call.

**A plan and its image.** A plan made for an image holds that image's size as `source`, and its
image `apply_` throws unless the view has exactly that size: `AffinePlan`, `FlipPlan` and
`LetterboxPlan` do. `MosaicPlan` holds a placement per patch and checks each against its patch.
`HSVPlan` holds three gains and fits any image.

**Errors.** `SA_THROW_IF` with `SushiAI::Error`. Every message opens with the name of the public
function that was called, a colon and a space: `apply_flip: `, `Image::reset: `,
`TrainingPipeline::make_sample: `, and `Image: ` for a constructor. A setting rejected through a
`plan_` function carries the `validate_` name. A size is printed as `{height}x{width}`. One
predicate, `is_valid_size`, holds the [1, `K_MAX_IMAGE_SIDE`] rule. The programme's later wave
turns these throws into status values.

## image.hpp

### ImageSize, is_valid_side, is_valid_size, is_valid_channel_count

Height first, then width, the order of the reference's `shape[:2]`. `K_MAX_IMAGE_SIDE` is 16384
and `K_MAX_CHANNELS` is 4. Both are this module's bounds, not the reference's. They let the raw
image reader refuse a header before it allocates. `K_RGB_CHANNELS` is 3.

### ImageView, Image

`ImageView` is a pointer and three extents with no ownership and no right to write. `Image` owns
a `std::vector<std::uint8_t>`. `Image::reset` changes the extents and keeps the allocation when
it is large enough; the pixel values afterwards are unspecified and every earlier view is
invalid, so a function that resets a destination writes every byte of it.
`Image::allocated_bytes` is the length of the allocation, which never shrinks. `data()` of an
image that never allocated is unspecified; test `allocated_bytes() == 0` instead.

### overlaps, check_source

`overlaps(view, image)` is true when the view's bytes and the image's allocation share a byte.
`check_source(source, destination, operation)` throws, in this order, when the source has no
pixels, a size `is_valid_size` rejects, a channel count outside [1, 4], or an overlap with the
destination.

## random.hpp

### Random

SplitMix64. The state is one 64-bit counter. A draw adds `K_INCREMENT` to the counter modulo
2^64 and returns `mix` of the sum:

```
z = value
z = (z ^ (z >> 30)) * K_FIRST_MULTIPLIER      (mod 2^64)
z = (z ^ (z >> 27)) * K_SECOND_MULTIPLIER     (mod 2^64)
mix = z ^ (z >> 31)
```

| Draw | Mapping | Draws consumed |
| --- | --- | --- |
| `next_u64()` | the 64 bits | 1 |
| `next_u32()` | `next_u64() >> 32` | 1 |
| `uniform()` | `(next_u64() >> 40) * 2^-24`, exact in `float`, in [0, 1) | 1 |
| `uniform(low, high)` | `low + (high - low) * uniform()` in `float`; a result that rounds to `high` becomes `std::nextafter(high, low)` | 1 |
| `index(count)` | `(uint64(next_u32()) * count) >> 32` | 1 |
| `bernoulli(p)` | `uniform() < p` | 1 |

Every mapping consumes exactly one draw whatever its arguments. The number of draws one sample
makes is not constant: it depends on whether the epoch is closed and on whether the mosaic gate
opened (7, 11 or 13; see the table under `pipeline.hpp`). That does not couple samples, because
`TrainingPipeline::make_sample` builds a generator of its own for each sample and no caller
passes one in. `index` rejects nothing; the probability of a value differs from `1 / count` by
at most 2^-32.

`derive(key)` returns a generator whose counter starts at `mix(state ^ mix(key + K_INCREMENT))`
and leaves the parent untouched. `make_sample_random(seed, epoch, index)` in `pipeline.hpp` is
the one place that applies it: `Random(seed).derive(epoch).derive(index)`.

The reference draws from Python's `random` and from `np.random` (Mersenne Twister, seeded per
worker at `data/build.py:233-235`). This module does not reproduce those sequences. A run here
and a run of the reference see different augmentations drawn from the same distributions.

## boxes.hpp

### to_pixel_box, to_normalized_box

`to_pixel_box` is the reference's `xywh2xyxy` followed by `denormalize`
(`utils/ops.py:279-284`, `utils/instance.py:282`): corners first, then the multiply by the image
size. `to_normalized_box` is `xyxy2xywh` followed by the division in `Format`
(`utils/ops.py:257-260`, `data/augment.py:2487-2488`).

### clip_box, box_area

`clip_box` clamps x to [0, width] and y to [0, height] (`utils/instance.py:426-427`).
`box_area` is `(x2 - x1) * (y2 - y1)` (`utils/instance.py:99`).

### CandidateFilter, make_polygon_filter, is_box_candidate

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

## polygon.hpp

### Point, Polygon

A polygon is the vertices of one object's outline in order, closed from the last vertex to the
first. An empty polygon means the object has none. The coordinates are fractions in a
`FileLabels` set and pixels in a `Labels` set.

### resampled_point_count, resample_polygon

The reference resamples every polygon of an image to 1000 points each time it loads the image,
or to the longest polygon's length plus one when that length is above 1000
(`data/dataset.py:416-422`). `resample_polygon` follows `resample_segments`
(`utils/ops.py:481-501`). With `m` the vertex count plus one (the first vertex appended) and
`n` the wanted count:

```
if vertex count == n:  unchanged
if m < n:   x = linspace(0, m - 1, n - m), then the integers 0 .. m - 1 merged in,
            each integer placed before the first x that is not below it
otherwise:  x = linspace(0, m - 1, n)
point(x) = vertex[j] + (vertex[j + 1] - vertex[j]) * (x - j),  j = floor(x)      (double)
```

`linspace(0, last, k)` is `i * (last / (k - 1))` for `i < k - 1` and `last` at the end, as
NumPy computes it. The result is cast to `float`. The resampled outline is the same closed
curve; it has more points so that the per-point clamp of the mosaic follows the canvas edge.

### map_polygon, clamp_polygon

`map_polygon` is `x * scale_x + offset_x`, `y * scale_y + offset_y` in `float`, multiply first.
Every transform moves a polygon through it. `clamp_polygon` clamps each vertex to a box.

### is_point_inside

Even-odd crossing count on a ray towards larger x, in `double`; a point on an edge or on a
vertex is inside. The reference calls `cv2.pointPolygonTest` here. OpenCV's source is not in
the Ultralytics package and was not read; the two can differ for a point within rounding of an
edge and for a self-intersecting outline.

### visible_polygon_box

The reference's `segment2box` (`utils/ops.py:78-124`), in `float`:

1. No vertex: four zeros.
2. Bounds inside [0, width] by [0, height]: the bounds.
3. Bounds wholly outside (`max < 0` or `min > extent` on an axis): four zeros.
4. Otherwise the bounds of three sets of points: the vertices inside the image; for each edge
   and each of the lines x = 0, x = width, y = 0, y = height, the point `start + t * delta`
   with `t = (bound - start) / delta`, kept when `t` is in [0, 1] and its other coordinate is
   inside the image; and each image corner that `is_point_inside` accepts. No point: four zeros.

An edge parallel to a line is skipped; the reference divides by zero there and discards the
result.

## label_set.hpp

### LabelSet, FileLabels, Labels

`LabelSet<Box>` owns three lists of equal length: classes, boxes, polygons. The constructors
throw when the lengths differ, `add` appends to all three and `keep_if` removes from all three,
so no transform checks the lengths. `box(i)` and `polygon(i)` give one entry for writing; no
member changes one list's length alone.

### resample_polygons, to_pixel_labels, to_normalized_labels

`resample_polygons` applies `resample_polygon` to each polygon of one image with the count
`resampled_point_count` of the longest. `to_pixel_labels` applies `to_pixel_box` and multiplies
each vertex by the width and the height (`utils/instance.py:283-284`). A pipeline's output is
`Labels`; the code that builds loss targets calls `to_normalized_labels` when it wants the
reference's batch form.

### clip_labels, remove_empty_boxes

`clip_labels` applies `clip_box` and clamps every vertex to the image
(`utils/instance.py:426-429`). `remove_empty_boxes` keeps an object when its box area is above
zero (`utils/instance.py:449`).

## resize.hpp

### fit_long_side

The reference's training and validation load (`data/base.py:297-299`): with
`r = image_size / max(height, width)` in `double`, a ratio of exactly 1 leaves the size alone,
and otherwise each side is `min(ceil(side * r), image_size)`. A small image is enlarged; the
reference has no guard against that.

### resize_bilinear

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

### resize_nearest

Each destination pixel is a copy of one source pixel, so the function never produces a value the
source does not hold, which is what an image of class indices needs. Destination column `x` reads
source column `floor((x + 0.5) * source.width / width)`, computed in 64-bit integers as
`(2x + 1) * source.width / (2 * width)` and capped at the last column; rows likewise. The
function has no reference: the label pipeline of the reference carries polygons, not masks. It
throws through `check_source` and through `Image::reset`, which is where a size outside the
bounds of an image is refused. `Datasets::fit_mask` computes the same index with its own
kernel, because it writes into a region of a letterboxed mask.

### resize_long_side

`resize_bilinear` to `fit_long_side`, or a copy when the size does not change.

## gray.hpp

### expand_gray

One channel in, three equal channels out. A source with another channel count throws. The
pipelines resize a grey image first and expand it after, which gives the same bytes as the
other order because the resize treats each channel alone.

## letterbox.hpp

### LetterboxSettings

The reference's validation transform is `LetterBox(new_shape=(imgsz, imgsz), scaleup=False)`
(`data/dataset.py:339`) with the constructor's other defaults (`data/augment.py:1704-1710`):
centred, padded with 114, no stride rounding, no stretch. `scale_up` and `padding_value` are the
two that can be set here. `auto`, `scale_fill` and an off-centre letterbox are off by default
and are not written. The training pipeline uses no letterbox, in the reference or here.

### plan_letterbox

`data/augment.py:1791-1811`:

```
ratio   = min(target.height / source.height, target.width / source.width)     (double)
ratio   = min(ratio, 1.0) unless scale_up
resized = (round_half_even(source.height * ratio), round_half_even(source.width * ratio))
dw      = target.width - resized.width          dh = target.height - resized.height
left    = floor(dw / 2)     right  = dw - left
top     = floor(dh / 2)     bottom = dh - top
```

The reference writes the split as `round(dw / 2 - 0.1)` and `round(dw / 2 + 0.1)` with Python's
`round`. For an integer `dw` those are `floor(dw / 2)` and `ceil(dw / 2)`, which is what the
lines above compute. `round_half_even` is Python's `round`.

### apply_letterbox

Image: the target is filled with `padding_value` outside the content rectangle, and the
content is the source copied, or resized by the kernel of `resize_bilinear` straight into rows
`top` onwards at column `left` when `plan.resized` differs from the source size
(`data/augment.py:1838-1849`). No scratch image is allocated.

Labels: `x * ratio + left`, `y * ratio + top` for boxes and vertices (`data/augment.py:1931-1932`,
called with `padw = left`, `padh = top` at `1893`). No clip and no filter, as in the reference.

## flip.hpp

### FlipSettings, validate_flip, plan_flip

`fliplr` is 0.5 (`cfg/default.yaml:124`). The reference decides with `random.random() < self.p`
(`data/augment.py:1588`); `plan_flip` decides with `bernoulli(probability)`, the same test. The
vertical flip is off by default (`flipud` is 0.0, `cfg/default.yaml:123`) and is not written.
The reference still draws for it; this module does not.

### apply_flip

Image: `np.fliplr` (`data/augment.py:1610`). Labels: the reference flips the centre of an xywh
box, `x_center = w - x_center`, and every vertex, `x = w - x` (`utils/instance.py:377-378`). On
corners that is `x1 = w - x2` and `x2 = w - x1`; `w` is `plan.source.width`, 640 after the warp.

## hsv.hpp

### HSVSettings, validate_hsv, plan_hsv

Gains 0.015, 0.7 and 0.4 (`cfg/default.yaml:115-117`). The reference draws
`np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]` (`data/augment.py:1517`), three values in
the order hue, saturation, value; `plan_hsv` draws the same three in the same order. The
reference makes no draw when all three gains are zero; `plan_hsv` still makes its three, which
changes nothing for another sample.

### is_hsv_identity

True when the plan's three gains are all zero. The reference skips the whole transform when the
three settings are zero (`data/augment.py:1514`); three zero settings always give such a plan,
and `apply_hsv` then copies the image instead of sending it through the hexcone round trip,
which would move a level by up to 4. A plan from non-zero settings is an identity only when all
three draws are exactly zero.

### make_hsv_tables

`data/augment.py:1518-1523`, in `double` as in the reference:

```
hue[x]        = trunc(floored_mod(x + plan.hue * 180, 180))
saturation[x] = trunc(clamp(x * (1 + plan.saturation), 0, 255))
value[x]      = trunc(clamp(x * (1 + plan.value), 0, 255))
saturation[0] = 0
```

`floored_mod(a, 180)` is `a - 180 * floor(a / 180)`, the result of `%` on NumPy floats.

### to_hsv, to_rgb

The reference converts with `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` and back
(`data/augment.py:1525-1527`). The `% 180` in its hue table shows that hue is 8-bit in steps of
two degrees, 0 to 179. The two functions below are the hexcone model in integers, written for
this module.

`to_hsv`, with `r`, `g`, `b` the levels, `v = max(r, g, b)` and `d = v - min(r, g, b)`:

```
value      = v
saturation = 0                       if v == 0
           = (255 * d + v / 2) / v   otherwise                (integer division)
hue        = 0                       if d == 0
otherwise:
    n = 30 * (g - b)              if v == r
      = 30 * (b - r) + 60 * d     else if v == g
      = 30 * (r - g) + 120 * d    otherwise
    if n < 0: n += 180 * d
    hue = (2 * n + d) / (2 * d)                                (integer division)
    if hue >= 180: hue -= 180
```

`to_rgb`, with `h`, `s`, `v` the levels, `i = h / 30` and `f = h - 30 * i`:

```
p = (v * (255 - s) + 127) / 255
q = (v * (7650 - s * f) + 3825) / 7650
t = (v * (7650 - s * (30 - f)) + 3825) / 7650                  (integer division, 7650 = 255 * 30)

i = 0: (v, t, p)    i = 1: (q, v, p)    i = 2: (p, v, t)
i = 3: (p, q, v)    i = 4: (t, p, v)    i = 5: (v, p, q)        as (red, green, blue)
```

The round trip is not the identity, because hue has 180 steps: (12, 200, 99) comes back as
(12, 200, 100).

### apply_hsv

Per pixel: `to_hsv`, the three tables, `to_rgb`. The image is copied unchanged when it does not
have three channels (`data/augment.py:1512-1513`) or when `is_hsv_identity(plan)`.

## affine.hpp

### AffineSettings, validate_affine

`scale` 0.5 and `translate` 0.1 (`cfg/default.yaml:119-120`, `data/augment.py:1102-1103`).
Rotation, shear and perspective are 0.0 by default (`cfg/default.yaml:118`, `121`, `122`) and are
not written, so the plan holds a scale and a translation and no general matrix builder. The
border value 114 is the reference's `borderValue` (`data/augment.py:1212`). `filter` and
`polygon_filter` are the two candidate filters of `boxes.hpp`.

### plan_affine

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

### apply_affine, image

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

### apply_affine, labels

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

## affine_nearest.hpp

### apply_affine_nearest

The warp of `apply_affine` with one tap in place of four. Output pixel `(x, y)` reads the source
pixel at `((x - m[2]) / m[0], (y - m[5]) / m[4])`, each coordinate computed in `double` and
rounded half up, `floor(position + 0.5)`. A coordinate that rounds outside the source gives
`plan.border_value`. No two source values are blended, so an image of class indices holds only
indices of its source and the border value. Every channel of a pixel moves together.

It takes the `AffinePlan` that `plan_affine` made for the image, so a mask warped with it lies
under the image `apply_affine` warped: both read the same source position, one blends its four
neighbours and the other takes the nearest. The caller sets `border_value` to the byte that
names no class.

It returns a `Status` and throws nothing.

| Code | When |
| --- | --- |
| `InvalidDescription` | A source of no pixels or of a size or channel count outside the bounds; a source that shares pixels with the destination; an output size outside the bounds; matrix scales that are not finite, not above zero or not `plan.scale`; an off-diagonal term; a translation that is not finite |
| `Incompatible` | A source of another size than `plan.source` |
| `OutOfMemory` | An allocation fails |

## mask_pipeline.hpp

### MaskSource

The seam a mask is loaded through, as `SampleSource` is for an image: `load(index, mask)`
returns the mask of image `index` at the image's stored resolution, one channel, one class index
per pixel, or the status of what kept it from being read.

### TrainingMaskPipeline

`make_mask(index, trace, loaded_size, mask)` gives a mask the geometry of the sample
`TrainingPipeline::make_sample` made for `index`. It draws nothing; every decision comes from
the `TrainingTrace` and the settings.

1. `load(index)`, then `resize_nearest` to `fit_long_side` of the mask's size, the size the
   image was loaded at. The result must be `loaded_size`, which is `Sample::loaded_size`.
2. When `trace.mosaic.is_enabled`: load and resize the masks of the three partners the same
   way, `plan_mosaic` with the trace's draw and the four mask sizes, and `apply_mosaic` with
   the fill value replaced by the void value.
3. `apply_affine_nearest` with `trace.affine`, its border value replaced by the void value.
4. `apply_flip` with `trace.flip`.

The colour jitter is not applied: it moves no pixel. The void value is the constructor's third
argument; `vision/datasets` passes `K_MASK_VOID`, 255.

A partner's mask is placed by the size it has, which is its image's when the dataset is
consistent. Only the sample's own mask is compared with its image.

| Code | When |
| --- | --- |
| `Incompatible` | A mask of several channels or of no pixels; the sample's own mask loaded at another size than its image; a warped mask of another size than the flip was planned for |
| `InvalidDescription` | An `image_size` outside [1, 16384]; what `apply_affine_nearest` refuses |
| `OutOfMemory` | An allocation fails |
| Any other | What `MaskSource::load` returns |

The settings must be the ones the `TrainingPipeline` was constructed with. The calls into
`resize_nearest`, `plan_mosaic`, `apply_mosaic` and `apply_flip` run under `Detail::guard`,
which stops the process for an `Error`: after the checks above, one of them throwing is a
defect of this unit.

Both units are in `test_imaging_mask_pipeline.cpp`. The mosaic test gives image `i` one box
over all of it with the class `i` and a mask that is `i` everywhere, and expects the mask to
hold a box's class at every pixel at least 2 pixels inside the box.

## partner_source.hpp, dataset_partners.hpp, recent_partners.hpp

### PartnerSource

The seam with one job: `choose(index, random)` returns the three partner indices of one sample
from exactly three draws. `draw_mosaic` calls it once per sample, before the gate, so a source
that keeps state sees every sample of an open epoch. A training pipeline owns its source
through a `std::unique_ptr`.

### RecentPartners, the default

The reference's rule when images are not cached in RAM, which is its default (`cache: False`,
`cfg/default.yaml:19`; `data/augment.py:482`, `499-500`). The reference keeps a list of the
indices it loaded most recently and draws partners from it with `random.choices`.

- Length (`data/base.py:168`): `recent_capacity = min(dataset size, 8 * batch size, 1000)`.
- Update (`data/base.py:306-312`): an index not in the buffer is appended; then, when the
  buffer holds more than one entry and at least `capacity` of them, the oldest leaves. A full
  buffer therefore holds `capacity - 1` indices, and one index when the capacity is 1 or 2.
- An index already in the buffer is not appended again and does not move.
- Draw: three times `recent[random.index(recent.size())]`. The sample's own index is in the
  buffer by then, so the first sample of a pipeline has itself as all three partners.

`TrainingPipeline(source, settings)` builds this source from `source.size()` and
`settings.batch_size` (16, `cfg/default.yaml:15`).

### DatasetPartners

The reference's rule under a RAM cache (`data/augment.py:501-502`): three times
`random.index(dataset size)`. It keeps no state.

## mosaic.hpp

### MosaicSettings, validate_mosaic, validate_mosaic_size, draw_mosaic

`mosaic` is 1.0 (`cfg/default.yaml:126`); the canvas fill is 114 (`data/augment.py:611`).

`validate_mosaic_size` owns the rule that the 2s canvas is a valid image: `image_size` is in
[1, `K_MAX_MOSAIC_IMAGE_SIZE`], which is `K_MAX_IMAGE_SIDE / 2`. `draw_mosaic`, `plan_mosaic`,
both `apply_mosaic` overloads and `validate_training` call it, so its message opens with
`validate_mosaic_size:` whoever the caller is.

Draws, in order: the three partners through the `PartnerSource`, the gate, `center_y`,
`center_x`. The reference draws the gate first (`data/augment.py:340`) and the partners after
it; the order differs here so that the source is asked for every sample. The reference skips
the mosaic when `random.uniform(0, 1) > p`; `draw_mosaic` makes it when
`bernoulli(probability)`, which differs only on a draw equal to `p`.

Centre (`data/augment.py:480`, `520`): `center_y` first, then `center_x`, each
`int(uniform(b, 2s - b))` with `b = (s + 1) / 2`. For s = 640 that is a value in 320 to 959.

### plan_mosaic

`data/augment.py:525-538`. With centre `(cx, cy)`, patch size `(h, w)` and canvas side `c = 2s`:

| Patch | Canvas `x1, y1, x2, y2` | Patch `x1, y1, x2, y2` |
| --- | --- | --- |
| 0, top left | `max(cx - w, 0), max(cy - h, 0), cx, cy` | `w - (x2a - x1a), h - (y2a - y1a), w, h` |
| 1, top right | `cx, max(cy - h, 0), min(cx + w, c), cy` | `0, h - (y2a - y1a), min(w, x2a - x1a), h` |
| 2, bottom left | `max(cx - w, 0), cy, cx, min(c, cy + h)` | `w - (x2a - x1a), 0, w, min(y2a - y1a, h)` |
| 3, bottom right | `cx, cy, min(cx + w, c), min(c, cy + h)` | `0, 0, min(w, x2a - x1a), min(y2a - y1a, h)` |

`x2a - x1a` is the width of the canvas rectangle in the same row. `offset_x = canvas_x1 -
patch_x1` and `offset_y = canvas_y1 - patch_y1`. Patch 0 is the sample; patches 1 to 3 are the
partners in draw order. Each patch is an image as `resize_long_side` returned it; the mosaic
resamples nothing.

### apply_mosaic

Image: a canvas of 2s by 2s filled with `fill_value`, then the four copies
(`data/augment.py:611-617`). The canvas is not cropped; `apply_affine` reduces it to s by s.

Labels: each patch's boxes and vertices plus its offsets (`data/augment.py:724`,
`utils/instance.py:315-317`), joined in patch order (`773-774`), clipped to [0, 2s] with
`clip_labels` (`776`), then `remove_empty_boxes` (`777-778`). The clip clamps each vertex on its
own, which is why the polygons were resampled to 1000 points at load.

## labels.hpp

### parse_label_text

The function takes text, not a path. A label file that does not exist is a background image in
the reference (`data/utils.py:483-484`); the caller passes empty text for it. The mapping from an
image path to its label path (`data/utils.py:143-144`) is the caller's and is not declared here.

Rows and fields (`data/utils.py:443`): the text is split into lines, a line with no field is
dropped, and a row's fields are separated by whitespace. A line ends at `\n`, `\r`, `\r\n`,
`\v`, `\f`, 0x1C, 0x1D or 0x1E; a field ends at a space, a tab or 0x1F. The non-ASCII
separators Python also honours (U+0085, U+00A0, U+2028, U+2029) are not separators here and
make the row fail as "not a number".

| Check | Reference |
| --- | --- |
| Every field parses, whole, as a finite decimal number | `np.array(lb, dtype=np.float32)`, `data/utils.py:451` |
| If any row has more than 6 fields, the file is a polygon file | `data/utils.py:446` |
| In a polygon file every row has an odd field count of at least 7 | `data/utils.py:447`, `449` |
| Otherwise every row has exactly 5 fields | `data/utils.py:457` |
| Each of the four box values is at most 1.01 | `data/utils.py:460` |
| The class and each box value are at least -0.01 | `data/utils.py:461` |
| The class has no fractional part | `data/utils.py:462` |
| The class is below `class_count` | `data/utils.py:465-466` |

A polygon row is `class x1 y1 x2 y2 ...`. Its box is the bounds of its points,
`x_center = (min_x + max_x) / 2` and `width = max_x - min_x` in `float`
(`utils/ops.py:465-478`, `257-260`), and its points are kept, in file order, as the object's
polygon (`data/utils.py:446-450`). The coordinate checks apply to the box, not to the points,
as in the reference. Values inside the tolerance are stored as they are; nothing is clipped.
The points are not resampled here: `resample_polygons` does that on each load, because 1000
points for every object of a dataset would not fit in memory.

Any failed check throws `Error`. The reference drops the image and counts it as corrupt
(`data/utils.py:492-495`); the caller that builds a dataset decides the same here by catching.

Three rules are narrower or simpler than the reference:

- The reference accepts a polygon row of one point (3 fields) in a polygon file. This parser
  requires three points.
- A nonzero field whose magnitude is below the least normal `double` is rejected. The reference
  accepts it as a subnormal or as zero. The rule keeps the outcome the same on every standard
  library; the message differs ("below the smallest normal number" where `std::from_chars`
  returns the subnormal, "not a number" where it reports out of range).
- Duplicates. The reference removes repeated rows (`data/utils.py:470-475`) and, when it removes
  any, returns the survivors in `np.unique`'s sorted order. This parser keeps the first
  occurrence of each row and the file's order. A box row repeats another when all five values
  are equal; a polygon row, when the class and every point are equal. Only the order of the
  objects inside one image differs.

## raw_cache.hpp

One file holds one decoded image. A separate tool decodes the dataset once and writes these
files; the library reads them and has no image decoder. The reference's counterpart is one
`.npy` per image at the original resolution (`data/base.py:337-345`). The tool is not part of
this module and is not written.

### Byte layout, version 1

Every multi-byte field is little-endian. The header is 24 bytes; no field is padded.

| Offset | Size | Field | Value |
| --- | --- | --- | --- |
| 0 | 4 | magic | the bytes `S`, `A`, `I`, `M` (`K_RAW_IMAGE_MAGIC` as a word) |
| 4 | 4 | `format_version` | unsigned, 1 |
| 8 | 4 | `height` | unsigned, in [1, 16384] |
| 12 | 4 | `width` | unsigned, in [1, 16384] |
| 16 | 4 | `channels` | unsigned, 1 or 3 |
| 20 | 4 | `channel_order` | unsigned: 0 `Gray`, 1 `RGB`, 2 `BGR` |
| 24 | height * width * channels | pixels | `uint8`, row-major, channels innermost |

`channels` is 1 exactly when `channel_order` is `Gray`. The file ends with the last pixel byte.
The extension is `.saim`.

| Version | Release | Change |
| --- | --- | --- |
| 1 | not released; planned for 0.1.0 | The layout above |

### Reader and writer

The writer serialises field by field and writes only the current version. The reader refuses a
version it does not know and names both versions in the error; there is no older version to
upgrade from yet.

`read_raw_image_header` allocates nothing and rejects every field outside its range.
`read_raw_image_file` also compares the file size with `24 + height * width * channels` before
it resets the destination, so a truncated or padded file costs no allocation. The stream form,
`read_raw_image_pixels`, cannot know a stream's length; its allocation is bounded by
`K_MAX_IMAGE_SIDE` and it throws when the stream ends early.

The reader returns the module's order. A `BGR` file has channels 0 and 2 swapped as it is read,
so a cache written from OpenCV-ordered pixels is usable without rewriting it. A `Gray` file is
read as one channel and stays one channel in the format; the detection pipelines expand it.

## rectangle.hpp

### RectangleSettings, validate_rectangle

The reference builds its validation dataset with `rect=True` (`models/yolo/detect/train.py:76`).
The defaults are that call's: `image_size` 640, `stride` 32 (`max(model stride, 32)`,
`train.py:75`), `pad` 0.5 (`data/build.py:341`), and a batch of 32, twice the training batch
(`engine/trainer.py:317`).

### plan_rectangular_batches

`set_rectangle` (`data/base.py:405-427`). From the stored size of every image:

```
ratio[i] = height[i] / width[i]                                   (double)
order    = the indices sorted by rising ratio, equal ratios in index order
batch b  = order[b * batch_size .. (b + 1) * batch_size)
lo, hi   = the smallest and the largest ratio of batch b
unit     = (hi, 1)       if hi < 1
         = (1, 1 / lo)   else if lo > 1
         = (1, 1)        otherwise                                 (height, width)
shape[b] = ceil(unit * image_size / stride + pad) * stride         per side
```

The result is `order` and one `shape` per batch; `shape_at(position)` is the target to pass to
`ValidationPipeline::make_sample` for the image `order[position]`. With the defaults, a batch
of images 480 high and 640 wide gets a shape 512 high and 672 wide.

## pipeline.hpp

### SampleSource

The seam between a pipeline and a dataset: a count, and a `load` that returns one decoded image
at its stored resolution, with one or three channels, and its `FileLabels`. A source over a
directory of raw image files and label files is not declared in this module.

### TrainingSettings, validate_training

`image_size` (640), `epochs` (100) and `close_mosaic` (10) come from `cfg/default.yaml:16`, `12`
and `34`; `batch_size` (16) sizes the default partner source; `seed` (0, `cfg/default.yaml:29`)
is the root of every sample's generator. The constructor calls `validate_training`, which passes
`image_size` to `validate_mosaic_size`, checks the other fields and then calls
`validate_mosaic`, `validate_affine`, `validate_hsv` and `validate_flip`.

### make_sample_random

`Random(seed).derive(epoch).derive(index)`. `make_sample` takes the sample's identity, not a
generator, so two samples cannot share a sequence and the order in which threads produce
samples cannot change what a sample draws.

### TrainingPipeline

The reference's order with its defaults is Mosaic, CopyPaste, RandomPerspective, MixUp, CutMix,
Albumentations, RandomHSV, vertical RandomFlip, horizontal RandomFlip, Format
(`data/augment.py:2870-2912`, `data/dataset.py:331-353`). CopyPaste, MixUp, CutMix and the
vertical flip have probability 0. Albumentations runs only when that package is installed, and
it is not installed in the reference environment. `make_sample(index, epoch, sample)` is the
rest:

1. `source.load(index)`, `resize_long_side` to `image_size`, `expand_gray` when the image has
   one channel, `resample_polygons`, `to_pixel_labels` with the loaded size
   (`data/base.py:445`, `data/dataset.py:416-422`).
2. Unless `is_mosaic_closed(epoch, epochs, close_mosaic)`: `draw_mosaic`. When the draw is
   enabled, load the three partners the same way, `plan_mosaic`, `apply_mosaic`.
3. `plan_affine` from the canvas, or from the loaded image when step 2 made no mosaic, to
   `image_size` by `image_size`; `apply_affine`.
4. `plan_hsv`, `apply_hsv`.
5. `plan_flip`, `apply_flip`.

The sample is an `image_size` by `image_size` RGB image and `Labels` in its pixels.
`original_size` and `loaded_size` describe the image of `index`, patch 0 of a mosaic, as the
reference's `ori_shape` does (`data/augment.py:771`). The polygons stay in the sample's labels;
the reference's detection `Format` drops them.

Draws per sample, in order:

| # | Function | Draw |
| --- | --- | --- |
| 1-3 | `draw_mosaic`, through the partner source | three partner indices |
| 4 | `draw_mosaic` | gate |
| 5-6 | `draw_mosaic` | `center_y`, `center_x`, only when the gate passed |
| 7-9 | `plan_affine` | scale, translate x, translate y |
| 10-12 | `plan_hsv` | hue, saturation, value |
| 13 | `plan_flip` | flip |

A mosaic sample makes 13 draws, a sample whose gate stayed shut 11, and a sample of a closed
epoch 7, because `draw_mosaic` is not called there. The reference still draws its gate in a
closed epoch and draws for every transform whose probability is 0; this module does neither.

With `DatasetPartners` a sample is a function of the seed, the epoch and the index alone. With
`RecentPartners`, the default, its three partners also depend on which samples the same
pipeline made before it, as they do in each worker process of the reference; its other draws
do not.

### is_mosaic_closed

`epoch >= epochs - close_mosaic` with a zero-based epoch (`engine/trainer.py:505`);
`close_mosaic` is 10 and `epochs` is 100 (`cfg/default.yaml:34`, `12`). The reference closes the
mosaic by setting its probability to 0 and rebuilding the transforms
(`data/dataset.py:390-394`), so the affine warp, the jitter and the flip stay on, as they do here.

### ValidationPipeline

`source.load(index)`, `resize_long_side`, `expand_gray` when needed, `resample_polygons`,
`to_pixel_labels`, `plan_letterbox` to `target`, `apply_letterbox` (`data/dataset.py:339`). It
returns the plan: `ratio`, `left` and `top` are what the reference keeps as `ratio_pad`
(`data/augment.py:1895-1900`) to map a prediction back to the original image.

`target` is a parameter. A square validation passes `image_size` by `image_size`; a rectangular
one passes `plan_rectangular_batches(...).shape_at(position)`. Which of the two a run uses is
the batch-shape half of decision D7 (SPEC section 8), which is still open.

## Known differences from the reference

- **Random sequences.** A different generator, fewer draws and another draw order inside the
  mosaic; see `random.hpp` and the draw table above.
- **Resize.** OpenCV's `INTER_LINEAR` arithmetic is not in the Ultralytics package and was not
  read: its coefficient width and its rounding are unknown here. This kernel uses 11-bit
  weights, rounds each weight to nearest and rounds the blend once. The pixel-centre mapping is
  the one OpenCV documents. Like OpenCV's, the kernel is two taps per axis at every scale, so a
  strong reduction aliases. The difference in output levels has not been measured.
- **Warp.** OpenCV's `warpAffine` inverts `M` and steps source coordinates in fixed point; that
  code was not read. This kernel divides in `double` per output coordinate and uses the 11-bit
  weights of the resize. A neighbour outside the source is the border value here; OpenCV's
  constant-border handling at the edge was not compared. The difference in output levels has
  not been measured.
- **HSV.** OpenCV's 8-bit `cvtColor` was not read. `to_hsv` and `to_rgb` are the hexcone model
  in integers with round-half-up divisions; level-for-level agreement with OpenCV has not been
  measured. A zero plan copies the image here, as the reference's zero settings skip the
  transform, but `plan_hsv` draws where the reference does not.
- **Point in polygon.** `is_point_inside` replaces `cv2.pointPolygonTest`, unread; see
  `polygon.hpp`.
- **Polygon point counts in a mosaic.** When the four patches' polygons have different point
  counts (one image has a polygon of more than 1000 vertices), the reference resamples them all
  to the longest (`utils/instance.py:505-516`). This module joins them as they are; the outlines
  are the same curves.
- **Polygon threshold in a mixed mosaic.** Per object here, per image in the reference; see
  `boxes.hpp`.
- **Mosaic partner buffer.** The reference appends an index when it loads the image from disk,
  mosaic or not, and keeps the buffered images in memory. Here the buffer holds indices only,
  the three partners are loaded again through the `SampleSource`, and the buffer is updated by
  `draw_mosaic`, so a closed epoch does not update it. No mosaic follows a closed epoch in one
  run, so that last point changes no sample.
- **Rectangular batch order.** NumPy's `argsort` is not stable by default, so the reference's
  order among images of equal aspect ratio depends on its sort routine. This module keeps
  equal ratios in index order. The batch shapes are the same either way.
- **Grey images.** Expanded after the resize, not decoded as colour; the bytes are the same.
- **Box order after a duplicate.** File order, not sorted order.
- **Nothing in this module has run.** Every expected value in the tests comes from hand
  arithmetic over the formulas above, except the polygon cases in `test_imaging_polygon.cpp`,
  `test_imaging_affine.cpp` and the shapes in `test_imaging_rectangle.cpp`, which were printed
  by the reference package's `resample_segments` and `segment2box` and by the lines of
  `set_rectangle` run under NumPy.
