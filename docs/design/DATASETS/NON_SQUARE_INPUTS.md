# Non-square inputs

Phase NS1 of [DATASETS](README.md).

## Problem

A model whose input is not square, such as ByteTrack's YOLOX-X at 800 by 1440, ran inference
through the letterbox and the plain resize, which already take a height and a width. The
training transforms did not. `AugmentOptions`, `TrainingSettings` and `ValidationSettings` held
one side, the mosaic canvas was `2s` by `2s`, the load fitted the long side to `s`, and the
`default` row of the pipeline table refused an input that was not square. The Ultralytics
importer wrote a square input and divided both sides of an anchor by one side, and an `imgsz`
list in a checkpoint fell back to 640 without a word.

## Decision

The size a transform produces is an `Imaging::ImageSize`, height and width, everywhere a side
stood. A square is the special case, not a separate path, and it reproduces the reference to the
bit.

- **The load fits within the size.** `fit_within(original, target)` scales by
  `min(target.height / height, target.width / width)` and caps each side at its target.
  `resize_within` resizes to it. For a square target the smaller ratio is `s / max(h, w)`
  exactly, so the reference's `data/base.py:297-299` rule is kept.
- **The mosaic canvas is twice each side.** `mosaic_canvas(size)` is `2h` by `2w`. The centre
  row is drawn from the height and the column from the width, each by the reference's rule
  `int(uniform(b, 2s - b))`, `b = (s + 1) / 2`, in the reference's order: row, then column. A
  square size makes the same draws as before.
- **The warp already took an output size**, so it reduces the canvas to `h` by `w` unchanged.
- **The pipelines carry the size.** `TrainingSettings::image_size`,
  `ValidationSettings::image_size` and `AugmentOptions::image_size` are `ImageSize`;
  `validate_mosaic_size` checks each side against `K_MAX_MOSAIC_IMAGE_SIZE`. The `default` row
  takes the input size as it is. `square_size(side)` builds the square case.
- **Rectangular batches stay long-side.** `RectangleSettings::image_size` is the long side the
  reference's `set_rectangle` scales every batch shape from; its shapes are rectangles already,
  so it keeps one side.
- **The sample cache keeps a longest side.** `longest_side` caps what is stored, not what a
  model reads, and is applied as `fit_within` of a square.
- **The Ultralytics importer takes an `UltralyticsInputSize`.** The input shape is written
  height first, and an anchor stated in pixels becomes its width over the input width and its
  height over the input height, which is what the anchor task reads. A checkpoint's `imgsz` is
  an integer or a list of one or two sides; Ultralytics' trainer trains a list at a square of
  its largest side, so the import does the same, and any other kind is refused.

## Not in this phase

The Darknet forward cases keep one side per model: they describe square reference networks, and
their `side` is a fact of those fixtures, not a limit of the library.
