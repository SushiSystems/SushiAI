# imaging

The host-side data pipeline of the YOLO26 programme: label parsing, the raw image file, resize,
letterbox, mosaic, random affine, HSV jitter and horizontal flip, composed into a training
pipeline and a validation pipeline. Everything works on `uint8` pixels and `float` boxes in host
memory. Nothing here reaches an engine, SYCL or a third-party library, so every unit is tested
without a device. The module depends on `failure`, `seam` for `Error` and `SA_THROW_IF` and on the
C++
standard library. `affine_nearest.hpp` and `mask_pipeline.hpp` return a `Status` and include
`<SushiAI/failure/status.hpp>`, so `failure`, `seam` has to become a public dependency;
`modules/vision/imaging/CMakeLists.txt` still declares it private.

Each header has one source under `modules/vision/imaging/source/` and one test file under
`tests/vision/imaging/`, named `test_<header>.cpp` and compiled into
`sushiai_vision_imaging_tests`. `partner_source.hpp` is an interface with no source and no test.
`test_mask_pipeline.cpp` tests `affine_nearest.hpp` and `mask_pipeline.hpp`.
`modules/vision/imaging/source/bilinear.hpp` and `bilinear.cpp` are private and hold the
fixed-point arithmetic the resize, the letterbox and the warp share.

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

[`include/SushiAI/vision/imaging/image.md`](include/SushiAI/vision/imaging/image.md) describes it.

## random.hpp

[`include/SushiAI/vision/imaging/random.md`](include/SushiAI/vision/imaging/random.md) describes it.

## boxes.hpp

[`include/SushiAI/vision/imaging/boxes.md`](include/SushiAI/vision/imaging/boxes.md) describes it.

## polygon.hpp

[`include/SushiAI/vision/imaging/polygon.md`](include/SushiAI/vision/imaging/polygon.md) describes it.

## label_set.hpp

[`include/SushiAI/vision/imaging/label_set.md`](include/SushiAI/vision/imaging/label_set.md) describes it.

## resize.hpp

[`include/SushiAI/vision/imaging/resize.md`](include/SushiAI/vision/imaging/resize.md) describes it.

## gray.hpp

[`include/SushiAI/vision/imaging/gray.md`](include/SushiAI/vision/imaging/gray.md) describes it.

## letterbox.hpp

[`include/SushiAI/vision/imaging/letterbox.md`](include/SushiAI/vision/imaging/letterbox.md) describes it.

## flip.hpp

[`include/SushiAI/vision/imaging/flip.md`](include/SushiAI/vision/imaging/flip.md) describes it.

## hsv.hpp

[`include/SushiAI/vision/imaging/hsv.md`](include/SushiAI/vision/imaging/hsv.md) describes it.

## affine.hpp

[`include/SushiAI/vision/imaging/affine.md`](include/SushiAI/vision/imaging/affine.md) describes it.

## affine_nearest.hpp

[`include/SushiAI/vision/imaging/affine_nearest.md`](include/SushiAI/vision/imaging/affine_nearest.md) describes it.

## mask_pipeline.hpp

[`include/SushiAI/vision/imaging/mask_pipeline.md`](include/SushiAI/vision/imaging/mask_pipeline.md) describes it.

## partner_source.hpp, dataset_partners.hpp, recent_partners.hpp

[`include/SushiAI/vision/imaging/partner_source.md`](include/SushiAI/vision/imaging/partner_source.md) describes it.

## mosaic.hpp

[`include/SushiAI/vision/imaging/mosaic.md`](include/SushiAI/vision/imaging/mosaic.md) describes it.

## labels.hpp

[`include/SushiAI/vision/imaging/labels.md`](include/SushiAI/vision/imaging/labels.md) describes it.

## raw_cache.hpp

[`include/SushiAI/vision/imaging/raw_cache.md`](include/SushiAI/vision/imaging/raw_cache.md) describes it.

## rectangle.hpp

[`include/SushiAI/vision/imaging/rectangle.md`](include/SushiAI/vision/imaging/rectangle.md) describes it.

## pipeline.hpp

[`include/SushiAI/vision/imaging/pipeline.md`](include/SushiAI/vision/imaging/pipeline.md) describes it.

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
