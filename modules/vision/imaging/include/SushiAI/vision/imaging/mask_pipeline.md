# modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp


## MaskSource

The seam a mask is loaded through, as `SampleSource` is for an image: `load(index, mask)`
returns the mask of image `index` at the image's stored resolution, one channel, one class index
per pixel, or the status of what kept it from being read.

## TrainingMaskPipeline

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
