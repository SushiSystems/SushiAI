# modules/vision/datasets/include/SushiAI/vision/datasets/collator.hpp


| Member | Does |
| --- | --- |
| `task()` | Returns the task's name as a training file spells it |
| `targets()` | Returns one `Data::TargetDescription` per role |
| `collate(list, samples, batch)` | Writes `batch.inputs` and every role; `list` is read for file names |

`batch` comes from `Data::allocate_batch` over `targets()`. A batch that lacks a role, or holds
it with another element type or count, is `Incompatible`. A sample the task cannot use is
`InvalidDescription`, and the message opens with the sample's file in quotes. Every collator of
images writes the inputs through `write_batch_inputs`; `RegressionCollator` copies
`FittedSample::features` into inputs of `[N, F]` float32.

| Collator | `make` | Role | Shape | Type | `classes` |
| --- | --- | --- | --- | --- | --- |
| `ClassificationCollator` | `class_count` | `labels` | `[N]` | int32 | `class_count` |
| `DetectionCollator` | `class_count`, `max_objects` | `boxes` | `[N, M, 4]` | float32 | 0 |
| | | `classes` | `[N, M]` | int32 | `class_count` |
| | | `valid` | `[N, M]` | float32 | 0 |
| `BinaryCollator` | `positive_class` | `labels` | `[N]` | float32 | 0 |
| `MultiLabelCollator` | `class_count` | `labels` | `[N, C]` | float32 | 0 |
| `RegressionCollator` | `feature_count`, `target_count` | `values` | `[N, K]` | float32 | 0 |
| `SegmentationCollator` | `SegmentationOptions` | `mask` | `[N, H, W]` | int32 | `class_count` |

`BinaryCollator` reads a set of two classes and writes 1 for a sample whose `class_index` is
`positive_class` and 0 for the other. A sample without a class index, or with one outside
[0, 2), fails.

`MultiLabelCollator` writes row `n` as `C` flags: 1 for every class one of the sample's objects
has, and for its `class_index` when it has one; 0 elsewhere. A sample with neither has a row of
zeros. The objects are the ones `fit_sample` kept, so a class whose only box has no area after
the fit is not flagged. A class outside [0, `C`) fails.

`RegressionCollator` copies the `F` features of sample `n` into row `n` of the inputs and its
`K` targets into row `n` of `values`. A sample that does not hold `F` and `K` numbers fails, as
a sample of an image does.

`SegmentationCollator` writes pixel `(y, x)` of `FittedSample::mask` into `mask[n, y, x]`. A
byte below `class_count` is a class. The byte `K_MASK_VOID`, 255, which is also the padding of
`fit_mask`, becomes `ignore_index`, 255 by default. Any other byte fails with the file, the row
and the column. `class_count` is in [1, 255], because a mask holds bytes and 255 names no class;
`ignore_index` is any int32 outside [0, `class_count`). A sample with no mask of the input size
fails.

`SegmentationOptions::origin` selects where that mask comes from. `MaskOrigin::File`, the
default, reads `FittedSample::mask`. `MaskOrigin::Polygons` paints the polygons of
`FittedSample::objects` with `paint_polygon_mask`, after the fit, so the vertices are already in
pixels of the input and nothing is resampled. `background` is the class of a pixel no polygon
holds; left out, such a pixel is `K_MASK_VOID` and becomes `ignore_index`. The padding of a
letterbox and the border of a warp hold no polygon, so they take the background too. A
`background` that is no class, or one stated with `MaskOrigin::File`, is `InvalidDescription`.

`ClassificationCollator` copies `class_index`. A sample without one, or with one outside
[0, `class_count`), fails.

`DetectionCollator` writes object `m` of image `n` into slot `m`. A box is `x1, y1, x2, y2` in
pixels of the input, the units of `Detect::GroundTruthBatch`, which `AnchorTrainingStep::run`
takes. `valid` is 1 in a slot that holds an object and 0 in a padding slot, whose box and class
are 0. `M` is `max_objects`. An image with more than `M` objects fails and names its file; no
object is dropped. An object whose class is outside [0, `class_count`) fails the same way.

The role names are `K_ROLE_LABELS`, `K_ROLE_BOXES`, `K_ROLE_CLASSES`, `K_ROLE_VALID`,
`K_ROLE_VALUES` and `K_ROLE_MASK` of `<SushiAI/data/roles.hpp>`.
