# segment

## Owns

The host side of semantic segmentation: the mask painted from polygon outlines, the confusion
matrix of predicted against true classes, and the three numbers read from it. Everything is in
namespace `SushiAI::Vision::Segment`. Every unit is a function or a small class over
`SushiRuntime::span` or `Imaging::Image`; nothing here reaches an engine, a graph or a tensor, so
every unit is tested without a device. The intent is `docs/design/DATASETS.md`, section
"Segmentation".

The objectives that train a segmentation model are in `vision/training`
(`pixel_cross_entropy`, `dice`), the head is `SegmentHead` in `vision/blocks`, and the role
`mask` is written by `SegmentationCollator` in `vision/datasets`.

No unit is a port. The rules below are the module's own, and no reference implementation has
been run against them.

## Depends on

- `failure`, `seam`, for `Status`, `Result` and the failure macros (`PUBLIC_DEPENDS`). Of the core
  modules, `seam` is private (`PRIVATE_DEPENDS`).
- `vision/imaging`, for `Image`, `ImageSize`, `Point` and `Polygon` (`PUBLIC_DEPENDS`).
- `SushiRuntime`, for `span`.

`vision/training` depends on this module. `vision/datasets` does not, and this module does not
depend on it.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/segment/ignore_index.hpp>` | `K_DEFAULT_IGNORE_INDEX` |
| `<SushiAI/vision/segment/fill_outline.hpp>` | `fill_outline` |
| `<SushiAI/vision/segment/rasterise_outlines.hpp>` | `ClassOutline`, `rasterise_outlines` |
| `<SushiAI/vision/segment/confusion_matrix.hpp>` | `K_MAX_CONFUSION_CLASSES`, `ConfusionMatrix` |
| `<SushiAI/vision/segment/intersection_over_union.hpp>` | `class_occurs`, `class_iou`, `mean_iou` |
| `<SushiAI/vision/segment/pixel_accuracy.hpp>` | `pixel_accuracy` |
| `<SushiAI/vision/segment/argmax_classes.hpp>` | `argmax_classes` |

## Mask geometry

A mask is an `Imaging::Image` of one channel: one class index per pixel, in a byte, as
`Datasets::fit_mask` and `SegmentationCollator` read it.

### fill_outline.hpp

`fill_outline(parts, value, mask)` writes `value` to every pixel of an already sized mask whose
centre is inside the outline. Vertices are in pixels of the mask, with the convention of
`vision/imaging`: pixel `(x, y)` covers `[x, x + 1]` by `[y, y + 1]` and has its centre at
`(x + 0.5, y + 0.5)`.

The inside is decided by the even-odd rule over the edges of every part taken together. Two
parts that do not touch are both filled, and a part inside another cuts a hole in it. A
centre exactly on an edge is given to one side: an edge counts for a row when one of its ends
is at or above the row's centre and the other below, and a pixel is filled when its centre is
at or right of the left crossing and left of the right one. Two outlines that share an edge
therefore fill no pixel twice and leave none between them.

The fill is a scanline over the rows the outline spans. For each row it collects the
crossings of every edge, sorts them and fills between each pair, so the cost is rows times
edges, in `double`. A part with fewer than three vertices has no area and fills nothing. A
vertex outside the mask is allowed; the fill is clipped. A vertex that is not finite is
refused with its part and its position before any pixel is written.

### rasterise_outlines.hpp

`rasterise_outlines(outlines, size, background, mask)` resets the mask to `size`, fills it with
`background` and calls `fill_outline` for each `ClassOutline` in the order given. Where two
outlines overlap, the later one wins. A caller that wants the smaller object on top sorts the
outlines by area, largest first, before the call. A failure names the position of its outline.

Nothing calls the rasteriser yet. `read_coco_json` and `read_yolo_txt` keep one polygon per
object in a `LabelSet`; a collator that paints those polygons into the role `mask` is not
written.

### What stays in other modules

`Datasets::fit_mask` and `Datasets::load_sample_mask` were not moved here. `fit_mask` takes a
`Datasets::SampleFitOptions` and `load_sample_mask` a `Datasets::SampleList`, so this module
would have to depend on `vision/datasets` while `vision/datasets` called it, which is a cycle.
The nearest-neighbour resize of an index image is `Imaging::resize_nearest`.

## Metrics

### confusion_matrix.hpp

`ConfusionMatrix::make(classes)` builds `C * C` int64 counts, the row being the true class.
`add(predicted, truth, ignore_index)` counts each pixel whose true class is not the ignore
index, and `merge` adds another matrix, so a matrix kept across batches holds exact sums. A
predicted class is not checked where the pixel is ignored. `add` checks every pixel before it
counts one, so a refused batch leaves the matrix unchanged. The class count is at most
`K_MAX_CONFUSION_CLASSES`, 4096, which is a matrix of 128 MiB.

### intersection_over_union.hpp

With `n[t][p]` the count of true class `t` predicted as `p`:

```
class_iou(c)  = n[c][c] / (sum_p n[c][p] + sum_t n[t][c] - n[c][c])
class_occurs(c) = sum_p n[c][p] > 0
mean_iou      = mean of class_iou(c) over the classes for which class_occurs(c)
```

A class occurs when a counted pixel has it as its true class. A class that is predicted and
never true has an IoU of 0 and is left out of the mean; its pixels still lower the IoU of the
classes they were taken from. A matrix that counted no pixel gives 0.

### pixel_accuracy.hpp

`pixel_accuracy` is the sum of the diagonal over the counted pixels, and 0 for none.

### argmax_classes.hpp

`argmax_classes(scores, classes, picked)` writes the class of the largest of each pixel's
`classes` scores, the lowest class on a tie. It is a template over `float` and `double`.

## A training file

A model whose last row is a `segment_head` and whose output has the height and the width of its
input trains with this file:

```json
{
  "sushiai_train": 1,
  "model": "model.json",
  "objective": { "type": "pixel_cross_entropy", "ignore_index": 255 },
  "data":      { "reader": "mask_png", "task": "segmentation", "root": "data/ade",
                 "split": "train", "batch_size": 8, "input": [512, 512], "classes": 150,
                 "ignore_index": 255 },
  "optimizer": { "type": "adamw", "learning_rate": 1e-3, "weight_decay": 0.01 },
  "epochs": 30,
  "checkpoint": "segmenter.sushi"
}
```

`ignore_index` is stated twice because two parts read it: the collator writes it into the mask
where the file holds 255 or the letterbox padded, and the objective leaves out the pixels that
hold it. Both default to 255. `"type": "dice"` takes the same key. The file has not been run:
`applications/programs` is outside this work, and whether `train` hands a `reader` dataset to
`Train::Trainer` for these objectives is not checked.

## Tests

| File | Suite | Covers |
| --- | --- | --- |
| `tests/vision/segment/test_confusion_matrix.cpp` | `Small_SegmentConfusionMatrix` | Counts by truth and prediction, an ignored pixel, sums over batches, merge and clear, the refusals and that a refused batch counts nothing |
| `tests/vision/segment/test_intersection_over_union.cpp`, `test_pixel_accuracy.cpp`, `test_argmax_classes.cpp` | `Small_SegmentMetrics` | IoU per class and its mean by hand, a class that does not occur, pixel accuracy, an empty matrix, the class pick and its tie |
| `tests/vision/segment/test_rasterise_outlines.cpp` | `Small_SegmentRasterise` | A rectangle on pixel edges, the centre rule, a shared edge, a triangle, two parts, a hole, clipping, an outline without area, the later outline winning, the refusals |

Every expected value in these files was worked by hand. None has been run.
