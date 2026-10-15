# modules/vision/detect/include/SushiAI/vision/detect/average_precision.hpp


The validation metric: detections are matched to ground truth at ten IoU thresholds, the matches
of every image are accumulated, and one table comes out with precision, recall, AP50 and AP50-95
per class and their means. The reference computes this part in numpy float64, so the unit works
in `double`; only the IoU and the thresholds are float32. `Unit_AveragePrecision` in
`tests/vision/detect/test_average_precision.cpp` holds its tests. No fixture carries a
reference metric table, so those tests compare with values derived by hand.

## K_IOU_THRESHOLDS

The ten thresholds are the float32 values of `torch.linspace(0.5, 0.95, 10)`:

```
U/models/yolo/detect/val.py:59  self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
```

The header writes them as the decimal literals 0.5f to 0.95f, which round to the same float32
bits. `K_IOU_THRESHOLD_COUNT` is 10. `K_RECALL_POINTS` is 101 and `K_CONFIDENCE_POINTS` is 1000,
the lengths of the two `np.linspace` grids below. `K_METRIC_EPS` is `eps = 1e-16`, the default
argument of `ap_per_class`.

## match_detections

Takes one image: `[N, 6]` detection rows, `[M, 4]` ground-truth boxes in the same frame and `[M]`
classes. It writes `[N, K_IOU_THRESHOLD_COUNT]` flags, 1 where the detection is a true positive
at that threshold. The IoU is `pairwise_iou` of `iou.hpp`, as in the reference:

```
U/models/yolo/detect/val.py:393  iou = box_iou(batch["bboxes"], preds["bboxes"])
U/engine/validator.py:335        iou = iou * correct_class  # zero out the wrong classes
U/engine/validator.py:346        matches = np.nonzero(iou >= threshold)  # IoU >= threshold and classes match
U/engine/validator.py:350        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
U/engine/validator.py:351        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
U/engine/validator.py:352        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
U/engine/validator.py:353        correct[matches[:, 1].astype(int), i] = True
```

Line 351 keeps one pair per detection, the one of largest IoU. Line 352 then keeps one pair per
box, and because the pairs are by then in detection order, the box goes to its first detection.
The port does the same in two steps. Each detection first takes the box of its class with the
largest IoU that is at least 0.5; of two boxes with equal IoU it takes the later one. Then, at
each threshold, the detections are visited in row order and a detection whose IoU reaches the
threshold takes its box unless an earlier row holds it. Three consequences follow, and the tests
pin each:

- A detection whose best box is taken stays unmatched. It is not offered its second-best box.
- A box goes to the first detection that reaches the threshold, not to the one of largest IoU.
- A detection of another class never matches, whatever its overlap.

A detection whose IoU is not a number matches at no threshold. `iou >= threshold` is false for
NaN in the reference, and both comparisons of the port are written in that direction.

The function throws when the detection array is not a whole number of rows, when the box count
and the class count disagree, when a ground-truth class is negative, when a detection's score is
not finite, when a detection's class is not a whole number in `[0, 2^24)`, and when `correct`
does not hold ten values per detection. A box coordinate that is NaN or infinite is not refused:
such a row is matched to nothing and counts as a false positive, as in the reference.

Not ported: the `use_scipy` branch (`U/engine/validator.py:338-344`).

## integrate_precision

`compute_ap` with `method = "interp"`:

```
U/utils/metrics.py:782  mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
U/utils/metrics.py:783  mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
U/utils/metrics.py:786  mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
U/utils/metrics.py:791  x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
U/utils/metrics.py:793  ap = func(np.interp(x, mrec, mpre), x)  # integrate
```

The curve gets a point at recall 0 with precision 1 and two closing points with precision 0.
The envelope makes precision non-increasing from the right, it is sampled at 101 recall points,
and the 100 trapezoids are summed. The last sample is at recall 1, where the envelope is 0, so a
detector that finds every box scores 0.99 + 0.01 / 2 = 0.995 and not 1. An empty curve gives 0.5,
the area under the line from 1 to 0.

The function takes recall and precision after each detection in descending score and throws
when the two differ in length. It does not check that recall never decreases.

## AveragePrecision

`add_image` checks one image as `match_detections` does, matches it and stores the scores, the
classes and the flags of its detections, the classes of its boxes and the distinct classes the
image holds. A refused image leaves the accumulator as it was. The caller passes detections that
already went through the validation confidence and the per-image limit (`K_VALIDATION_CONFIDENCE`
and `K_MAX_DETECTIONS` of `detection_row.hpp`), in the frame of the boxes; the class applies
neither. `clear` drops every image and `image_count` returns how many were added.

`summarize` is `ap_per_class` over everything accumulated:

```
U/utils/metrics.py:843  i = np.argsort(-conf)
U/utils/metrics.py:847  unique_classes, nt = np.unique(target_cls, return_counts=True)
U/utils/metrics.py:868  recall = tpc / (n_l + eps)  # recall curve
U/utils/metrics.py:869  r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
U/utils/metrics.py:872  precision = tpc / (tpc + fpc)  # precision curve
U/utils/metrics.py:873  p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score
U/utils/metrics.py:884  f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
U/utils/metrics.py:892  i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
```

The table has one row per class that has ground truth, in ascending class index. A class with
detections and no ground truth gets no row and no share of a mean. A class with ground truth and
no detections gets a row of zeros, which enters every mean. With no ground truth at all the
table has no rows and the four means are 0.

For a class with both, the detections are ranked by descending score, recall and precision are
accumulated down the ranking at each threshold, and `integrate_precision` gives the AP of each.
`ap50` is the AP at IoU 0.5 and `ap50_95` is the mean of the ten. The recall and precision at
IoU 0.5 are also sampled at 1000 confidence values from 0 to 1.

Precision and recall are not read at a fixed confidence. The F1 curve of each class is averaged
over the rows, smoothed, and the confidence point where the result is largest is the one every
row is read at:

```
U/utils/metrics.py:670  nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
U/utils/metrics.py:672  yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
U/utils/metrics.py:673  return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
```

With 1000 points and `f = 0.1` the box filter is 101 points wide, and the curve is padded with
50 copies of its first and of its last value. Of equal maxima the first is taken, as `argmax`
does.

The means are those of the reference's `Metric`: `precision` and `recall` are the means of the
rows (`U/utils/metrics.py:963`, `972`), `map50` is the mean of the rows' AP at
IoU 0.5 (`U/utils/metrics.py:981`) and `map50_95` is the mean over rows and thresholds
(`U/utils/metrics.py:999`). `ClassMetrics::images` counts the images that hold a box of the
class, from `"target_img": np.unique(cls)` (`U/models/yolo/detect/val.py:258`), and `instances`
counts its boxes.

**Equal scores.** `np.argsort(-conf)` does not say in which order equal scores come out. The
port ranks detections of equal class and score in the order they were added, so its result for
such an input is defined and the reference's is not.

**Summation order.** The 100 trapezoids, the mean of a row's ten APs and the four means of the
table are summed in the order numpy's pairwise summation adds them: left to right below 8
values, eight lanes from 8 to 128 values, and a split in two above that. The F1 mean over the
rows and the taps of the box filter are summed left to right.

**Not ported.** The plots, and the true and false positive counts `ap_per_class` also returns.

**Cost.** The accumulator holds one float, one `int64` and ten bytes per detection, one `int64`
per box and one `int64` per distinct class of each image. `summarize` sorts the detections once,
and for each class with detections it runs ten passes over that class's detections and fills two
curves of 1000 points.
