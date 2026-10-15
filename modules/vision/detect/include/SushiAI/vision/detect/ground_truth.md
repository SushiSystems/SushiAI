# modules/vision/detect/include/SushiAI/vision/detect/ground_truth.hpp


## GroundTruthRows

The label rows of a batch as a data loader holds them: the image index, the class and the box of
each row, the box as centre x, centre y, width, height, each a fraction of the image side. These
are `batch_idx`, `cls` and `bboxes` of the reference batch (`U/utils/loss.py:431`).

## GroundTruthBatch

The same rows as `[B, M]` arrays, M being the largest box count of an image: labels, pixel
corner boxes and the 0/1 flag the reference calls `mask_gt`. The struct holds what the assigner
reads and nothing else. `pad_ground_truth` counts the rows of each image in a local `[B]` array,
one allocation of B integers per call.

## pad_ground_truth

`v8DetectionLoss.preprocess` and the two lines after its call:

```
U/utils/loss.py:394  out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
U/utils/loss.py:399  out[batch_idx, within_idx] = targets[:, 1:]
U/utils/loss.py:400  out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
U/utils/loss.py:432  ... scale_tensor=imgsz[[1, 0, 1, 0]])
U/utils/loss.py:434  mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
```

The four values are scaled by the image width and height and converted with `xywh_to_xyxy`.
A row is valid when the sum of its four pixel corners is above zero, as the reference derives it.
Rows of one image keep their order. The reference computes a row's slot as its index minus the
image's first index, which assumes rows grouped by image; this port counts per image, which gives
the same slots for grouped rows and stays defined for ungrouped ones. Labels are `int64` here and
float in the reference.
