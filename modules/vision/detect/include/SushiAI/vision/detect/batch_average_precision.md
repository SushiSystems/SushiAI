# modules/vision/detect/include/SushiAI/vision/detect/batch_average_precision.hpp


The feed between a batch and the metric: `add_batch` takes the detections of every image of a batch
and its `GroundTruthBatch` and calls `AveragePrecision::add_image` once per image. It is a SushiAI
unit with no reference counterpart; the validator's own loop does the same per image.

## add_batch

The truth of a batch is padded to `max_boxes` rows per image. `add_batch` copies the rows whose
`valid` flag is 1, in slot order, into the box and class arrays of one image, so a padding row never
reaches the metric as a box of class 0. The detections and the truth boxes must be in one frame; the
anchor report passes both in pixels of the network input. The function returns `InvalidDescription`
when the number of detection arrays is not `truth.batch`, or when an array of the truth does not
have `batch` times `max_boxes` rows, and adds nothing. The checks of `add_image` apply to each
image; an image whose `add_image` returns a failure leaves the images before it in the metric,
and `add_batch` returns that failure.

**Cost.** Two vectors hold one image's boxes and classes and keep their storage between images.
