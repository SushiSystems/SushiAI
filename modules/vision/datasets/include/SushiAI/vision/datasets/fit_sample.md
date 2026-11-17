# modules/vision/datasets/include/SushiAI/vision/datasets/fit_sample.hpp


`fit_sample` turns a loaded image and its `Sample` into a `FittedSample`: the pixels at the
input size, the objects as pixel boxes of that image, and the class index and file index copied.

| `SampleFit` | Image | Objects |
| --- | --- | --- |
| `Resize` | `Imaging::resize_bilinear` to the input size; each axis has its own scale | `Imaging::to_pixel_labels` on the input size |
| `Letterbox` | `Imaging::plan_letterbox` and `Imaging::apply_letterbox` with `SampleFitOptions::letterbox` | `Imaging::to_pixel_labels` on the source size, then `Imaging::apply_letterbox` with the same plan |
| `DarknetLetterbox` | `Imaging::plan_darknet_letterbox` and `Imaging::apply_darknet_letterbox`: truncated sides, offsets `(input - side) / 2` truncated, padding 128, scaled up and resized by `Imaging::resize_darknet`; `SampleFitOptions::letterbox` is not read | `Imaging::to_pixel_labels` on the source size, then `Imaging::apply_darknet_letterbox` with the same plan |

After each fit the objects go through `Imaging::clip_labels` and `Imaging::remove_empty_boxes`,
so a box lies inside the input and has an area above zero. The three fits are the table `K_FITS`
in `source/fit_sample.cpp`.

| Code | When |
| --- | --- |
| `InvalidDescription` | The input size is outside [1, `Imaging::K_MAX_IMAGE_SIDE`], or the fit names no row of the table |
| `OutOfMemory` | An allocation fails |
| Any other | A function of `vision/imaging` returns a status, such as `plan_letterbox` for a resized side of zero pixels; it is forwarded. A source that fails `Imaging::check_source` stops the process |
