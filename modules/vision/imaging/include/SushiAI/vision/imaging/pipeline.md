# modules/vision/imaging/include/SushiAI/vision/imaging/pipeline.hpp


## SampleSource

The seam between a pipeline and a dataset: a count, and a `load` that returns one decoded image
at its stored resolution, with one or three channels, and its `FileLabels`. A source over a
directory of raw image files and label files is not declared in this module.

## TrainingSettings, validate_training

`image_size` (640), `epochs` (100) and `close_mosaic` (10) come from `cfg/default.yaml:16`, `12`
and `34`; `batch_size` (16) sizes the default partner source; `seed` (0, `cfg/default.yaml:29`)
is the root of every sample's generator. The constructor calls `validate_training`, which passes
`image_size` to `validate_mosaic_size`, checks the other fields and then calls
`validate_mosaic`, `validate_affine`, `validate_hsv` and `validate_flip`.

## make_sample_random

`Random(seed).derive(epoch).derive(index)`. `make_sample` takes the sample's identity, not a
generator, so two samples cannot share a sequence and the order in which threads produce
samples cannot change what a sample draws.

## TrainingPipeline

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

## is_mosaic_closed

`epoch >= epochs - close_mosaic` with a zero-based epoch (`engine/trainer.py:505`);
`close_mosaic` is 10 and `epochs` is 100 (`cfg/default.yaml:34`, `12`). The reference closes the
mosaic by setting its probability to 0 and rebuilding the transforms
(`data/dataset.py:390-394`), so the affine warp, the jitter and the flip stay on, as they do here.

## ValidationPipeline

`source.load(index)`, `resize_long_side`, `expand_gray` when needed, `resample_polygons`,
`to_pixel_labels`, `plan_letterbox` to `target`, `apply_letterbox` (`data/dataset.py:339`). It
returns the plan: `ratio`, `left` and `top` are what the reference keeps as `ratio_pad`
(`data/augment.py:1895-1900`) to map a prediction back to the original image.

`target` is a parameter. A square validation passes `image_size` by `image_size`; a rectangular
one passes `plan_rectangular_batches(...).shape_at(position)`. Which of the two a run uses is
the batch-shape half of decision D7 (SPEC section 8), which is still open.
