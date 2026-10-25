# training

## Owns

The training of a detection model whose loss the graph cannot state as one expression: the loss
root that hands a host-computed delta to autograd, the objective of a model with anchor heads,
the two-pass step that computes the deltas between a forward-only run and the training
replay, and the copy of a batch's detection roles into the ground truth that step reads.
Everything is in namespace `SushiAI::Vision::Training`. The behaviour is in sections 4.6 to 4.8
of `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.

The delta itself and its reported cost belong to `vision/detect`, part `anchor/`
(`compute_head_delta`, `measure_head_cost`). This module only carries them into a graph.

It also owns the two objectives of semantic segmentation, `pixel_cross_entropy` and `dice`,
which a batch alone can feed and `Train::Trainer` therefore drives. Their intent is
`docs/design/DATASETS/README.md`, section "Segmentation".

## Depends on

- `train`, for `Objective`, `TrainingStep` and the graph builders.
- `vision/detect`, for the anchor layout, the head options, the delta, the cost and
  `GroundTruthBatch`.
- `vision/segment`, for the confusion matrix, the two metrics read from it, the class pick
  and `K_DEFAULT_IGNORE_INDEX`.
- `data`, for `Batch`, `TargetDescription` and the role names.
- `graph`, `nn`, `optim` and `failure`, `dtype`, `shape`, `seam`, `execution`, as `train` names
  them.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/training/head_delta_loss.hpp>` | `emit_head_delta_loss` |
| `<SushiAI/vision/training/distribution_expectation_graph.hpp>` | `emit_distribution_expectation` |
| `<SushiAI/vision/training/distribution_focal_loss.hpp>` | `emit_distribution_focal_loss` |
| `<SushiAI/vision/training/anchor_bias_init.hpp>` | `AnchorBiasHead`, `derive_anchor_bias_heads`, `initialize_anchor_biases` |
| `<SushiAI/vision/training/free_assignment_stage.hpp>` | `FreeLeafKind`, `FreeLeaf`, `FreeLevelViews`, `FreeAssignmentStage` |
| `<SushiAI/vision/training/free_batch_decoder.hpp>` | `FreeLevelLogits`, `FreeBatchDecoder` |
| `<SushiAI/vision/training/free_detection_loss.hpp>` | `FreeLossValues`, `FreeLossTermIDs`, `emit_free_detection_loss` |
| `<SushiAI/vision/training/free_detection_loss_host.hpp>` | `evaluate_free_detection_loss` |
| `<SushiAI/vision/training/free_loss_gains.hpp>` | `FreeLossGains`, `K_FREE_LOSS_GAIN_KEYS`, `validate_free_loss_gains`, `read_free_loss_gains` |
| `<SushiAI/vision/training/free_model_facts.hpp>` | `FreeLevelPorts`, `FreeLevelFacts`, `FreeModelFacts`, `derive_free_model_facts` |
| `<SushiAI/vision/training/free_bias_init.hpp>` | `free_class_bias`, `initialize_free_biases` |
| `<SushiAI/vision/training/anchor_head.hpp>` | `AnchorHead` |
| `<SushiAI/vision/training/head_delta_rule.hpp>` | `HeadDeltaBatch`, `HeadDeltaRule` |
| `<SushiAI/vision/training/head_delta_rule_choice.hpp>` | `choose_head_delta_rule` |
| `<SushiAI/vision/training/best_shape_head_delta_rule.hpp>` | `BestShapeHeadDeltaRule` |
| `<SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>` | `ShapeRatioHeadDeltaRule` |
| `<SushiAI/vision/training/anchor_objective.hpp>` | `AnchorHead`, `AnchorObjective` |
| `<SushiAI/vision/training/anchor_objective_factory.hpp>` | `build_anchor_objective`, `make_anchor_objective` |
| `<SushiAI/vision/training/anchor_training_step.hpp>` | `AnchorTrainingStep` |
| `<SushiAI/vision/training/anchor_training_loop.hpp>` | `AnchorProgress`, `AnchorProgressCallback`, `AnchorLoopOptions`, `AnchorRunReport`, `run_anchor_training` |
| `<SushiAI/vision/training/truth_from_roles.hpp>` | `K_TRUTH_BOX_VALUES`, `describe_truth_roles`, `truth_from_roles` |
| `<SushiAI/vision/training/pixel_extents.hpp>` | `PixelExtents` |
| `<SushiAI/vision/training/pixel_labels.hpp>` | `validate_pixel_labels` |
| `<SushiAI/vision/training/pixel_targets.hpp>` | `fill_pixel_targets` |
| `<SushiAI/vision/training/dice_targets.hpp>` | `DiceLeaves`, `fill_dice_targets` |
| `<SushiAI/vision/training/pixel_losses.hpp>` | `K_DICE_EPSILON`, `emit_pixel_cross_entropy`, `DiceOperands`, `emit_soft_dice` |
| `<SushiAI/vision/training/segmentation_metrics.hpp>` | `K_MEAN_IOU_METRIC`, `K_PIXEL_ACCURACY_METRIC`, `measure_segmentation` |
| `<SushiAI/vision/training/pixel_cross_entropy_objective.hpp>` | `K_PIXEL_CROSS_ENTROPY_TYPE`, `K_PIXEL_TARGETS_LEAF`, `K_PIXEL_WEIGHTS_LEAF`, `PixelCrossEntropyObjective`, `make_pixel_cross_entropy_objective` |
| `<SushiAI/vision/training/dice_objective.hpp>` | `K_DICE_TYPE`, the five `K_DICE_*_LEAF` names, `DiceObjective`, `make_dice_objective` |


`AnchorObjective` computes each batch's deltas and cost through a `HeadDeltaRule` chosen once from
the task's `assignment`: `best_shape` gives `BestShapeHeadDeltaRule`, which runs `compute_head_delta`
and `measure_head_cost` as before, and `shape_ratio` gives `ShapeRatioHeadDeltaRule`, which assigns
every level with `ShapeRatioAssignment` and returns `compute_shape_ratio_delta`'s deltas with the
shape-ratio loss as the cost. The shape-ratio rule takes logit heads whose `size` is
`squared_sigmoid` and refuses others; the Darknet rule takes `exponential` sizes only.
## The delta loss

`emit_head_delta_loss(builder, head, delta, scale)` emits `-scale * sum(delta * head)` as `MUL`,
`RESHAPE`, `SUM_ROWS` and `SCALE`. `delta` is an input leaf, so the gradient at `head` is
`-scale * delta` and the value of the root means nothing.

## The anchor objective

`AnchorObjective::make(network, layout, heads)` takes one `AnchorHead` per head of the layout, in
the layout's order: the name the model publishes the head under, its `HeadShape`, its
`HeadOptions` and its `HeadTrainingOptions`. The shape is part of the description because
`Train::Objective::declare` runs before the model is traced, when no head has a shape yet. Every
head carries the same batch.

- `declare` adds one float32 leaf per head, shaped `[B, H, W, A * (5 + C)]`, under the role
  `delta.<output name>`.
- `loss` adds `emit_head_delta_loss` over the heads with `scale = 1 / batch`. It throws, as the
  objectives of `train` do, when the model does not publish a head or publishes it in another
  shape.
- `measure_cost(deltas)` returns the `HeadCost` of every head summed field by field.
- It keeps the default `feed`, which reports that the objective is fed by its own step, and it
  reports no metric through `measure`; `measure_cost` reports the cost of its heads instead.

## The training step

`AnchorTrainingStep` owns one `Train::TrainingStep` over the graph of `build_training_graph`,
compiled around a `Train::HostStage` that reads every head and writes every delta leaf. The
step therefore has two parts: the nodes up to the last head, and the rest. The caller builds
the graph, binds its pool with `Train::bind_parameters`, makes the optimizer, and passes the
parts to `make`.

`run(inputs, truth)` does four things in order:

1. It checks the feed and the truth. A truth of another batch or array size is `Incompatible`; a
   valid row whose label names no class of a head is `InvalidDescription`.
2. It copies the inputs into the training graph and runs the first part, so the heads come out
   under batch statistics.
3. It calls `compute_head_delta` on each head and copies the delta into that head's leaf.
4. It runs the second part: the backward pass, the optimizer's update and the
   running-statistic blend, once.

The forward pass runs once a step, and the deltas come from the heads the backward pass
differentiates.

It returns `measure_cost` of the deltas it fed. `deltas()` returns them, and `output(name)`
gives the tensor of any output the model published, the heads among them.

The host stage reads the heads and writes the leaves through `Tensor::data_as<float>()`, so both
must be float32 and host-addressable, as the loss and the logits `Train::Trainer` reads are.

## The truth from roles

A detection dataset publishes three roles (`docs/design/DATASETS/README.md`): `boxes` `[N, M, 4]`
float32 as x1, y1, x2, y2 in pixels of the input, `classes` `[N, M]` int32, and `valid`
`[N, M]` float32. `AnchorTrainingStep::run` reads a `Detect::GroundTruthBatch`, which holds the
same rows as int64 labels and byte flags. `truth_from_roles.hpp` joins the two.

`describe_truth_roles(max_objects, classes)` returns the three `Data::TargetDescription`s in the
form `Train::match_roles` takes, so a program can refuse a dataset before the first batch. A
class count of 0 accepts a dataset of any count; the model's class count refuses one that
differs. An `M` below 1 and a negative class count are `InvalidDescription`.

`truth_from_roles(batch, truth)` finds the roles by name, in whatever order the batch holds
them, copies the corners, widens each class to int64 and writes each flag as a byte. The
caller owns `truth` and passes the same object for every batch: its arrays are assigned in
place, so after the first batch of a size the copy allocates nothing. The box units are left as
they are, because both sides use pixel corners.

| Refusal | Code |
| --- | --- |
| The batch lacks one of the three roles; the message lists the roles it carries | `InvalidDescription` |
| A flag that is neither 0 nor 1, a NaN included; the message names the image and the slot | `InvalidDescription` |
| A role of another rank or element type | `Incompatible` |
| A `boxes` whose last axis is not 4, or `classes` or `valid` with other `[N, M]` than `boxes` | `Incompatible` |
| A tensor with no storage or with strides that are not its layout's | `Incompatible` |

A failure leaves `truth` as it was. The function does not check a label against a class count
or a box against the image; `AnchorTrainingStep::run` and `validate_head_truth` do.

## Tests

| File | Covers |
| --- | --- |
| `tests/vision/training/test_head_delta_loss.cpp` | The four nodes, the root under autograd, the refusals |
| `tests/vision/training/test_anchor_objective.cpp` | The leaves, the summed root, the summed cost, the refusals |
| `tests/vision/training/test_anchor_objective_factory.cpp` | The heads from the published shapes, the options from the section, the refusals |
| `tests/vision/training/test_anchor_training_loop.cpp` | The rate of each update over two epochs, a run without a schedule, a run that starts at a later epoch and batch, the run count of both passes restored once, the refusals, a callback that throws |
| `tests/vision/training/test_truth_from_roles.cpp` | The three descriptions, the copy with a tolerance of zero, the storage kept between batches, the refusals |
| `tests/vision/training/test_pixel_targets.cpp` | The label check, the indices and weights of the cross entropy, the five Dice leaves, a mask with no kept pixel, the refusals and that a refused fill writes nothing. Needs an engine |
| `tests/vision/training/test_pixel_cross_entropy_objective.cpp`, `test_dice_objective.cpp`, `test_pixel_extents.cpp` | What `make` keeps and refuses, the INT32 bound at 54 and 55 images of 512 by 512 over 150 classes, the leaves, the nodes of both losses, both factories and what they refuse |
| `tests/vision/training/test_segmentation_metrics.cpp` | Mean IoU and pixel accuracy of one step by hand, and the steps refused. Needs an engine |
| `tests/vision/training/test_pixel_cross_entropy_objective.cpp` (root) | Both losses and their gradients at the logits against the host, a mask of ignored pixels alone, a label out of bounds |
| `tests/vision/training/test_pixel_cross_entropy_objective.cpp` (root) | Two convolutions and a `SegmentHead` fitting one batch through `Train::Trainer` |
| `tests/vision/training/test_head_delta_loss.cpp` (root) | The gradient at the head on an engine |
| `tests/vision/training/test_anchor_training_step.cpp` (root) | The leaves fed, a bias after one step, a running mean blended once, the refusals |

## The objective from the files

`build_anchor_objective(given, attributes)` builds an `AnchorObjective` from what a run's files
state. It reads three things of the `Train::ObjectiveContext`:

| From | Through | Gives |
| --- | --- | --- |
| `given.task` | `Detect::Anchor::read_anchor_task` | The layout, and each head's output name and decode options |
| `attributes` | `Detect::Anchor::read_head_training_section` | One `HeadTrainingOptions`, given to every head |
| `given.input_sample`, `[H, W, C]` | | The `NetworkSize` |
| `given.outputs` | `Train::find_output_shape` | Each head's `HeadShape`, from `[batch, H, W, A * (5 + C)]` |

It refuses a model with no task, a head the model does not publish, an output whose batch or
channels differ from what the task implies, and data whose class count is not the task's. A
class count of 0 in the context is data that states none and is accepted.
`make_anchor_objective` is the same function in the shape of a `Train::ObjectiveFactory`; it is
the `anchor` row `Vision::register_objectives` adds.

The head shapes come from `Train::list_output_shapes` over the probe graph, which
`AnchorTrainingStep` needs anyway and which is traced with no engine.

## The loop

`run_anchor_training(step, dataset, options)` drives an `AnchorTrainingStep` for
`options.epochs` passes. It is a second loop beside `Train::Trainer`, which stays the loop of
every objective that a batch alone can feed: `Trainer::run_step` asks the objective to fill its
leaves and reads a loss tensor, while an anchor update runs two passes and takes a
`GroundTruthBatch`.

Before the first update it checks the epochs, the batch count, the batch size against the
heads, and the schedule, then matches the roles of `describe_truth_roles` once, with the slot
count the dataset's `boxes` role states. For each batch it calls `Dataset::read`,
`truth_from_roles`, `Optim::apply_step_schedule` when `options.schedule` is set, and
`AnchorTrainingStep::run`, then hands an `AnchorProgress` to `options.on_update`. One `Batch`
and one `GroundTruthBatch` serve the whole run. A callback that throws ends the run as a failure.

A resumed run sets `options.first_epoch` and `options.first_batch`. The first pass calls
`Data::begin_epoch` with `first_epoch` and reads from `first_batch` on; each later pass takes
the next epoch index and starts at batch 0. `AnchorProgress::epoch` still counts the passes of
this call from 1. A negative `first_epoch` is `InvalidDescription` and a `first_batch` the
dataset lacks is `Incompatible`.

`AnchorTrainingStep::run_count()` returns the run count of the training pass, and
`restore_run_count(count)` sets the count of both passes, which `run` advances together. A
model with dropout draws a mask in each pass, so both need the count of the saved run.

## The per-pixel objectives

Both objectives read one NHWC `[N, H, W, C]` output, the logits, and the role `mask`, `[N, H, W]`
int32, in which a pixel holds a class in `[0, C)` or the ignore index. `R` below is `N * H * W`.
The logits are viewed as `[R, C]` rows by a `RESHAPE`; no value is copied. Both are made with a
`PixelExtents`, an ignore index, 255 by default, and the element type of their real leaves,
which must be the type of the logits. An ignore index inside `[0, C)` is refused.

`feed` checks every label with `validate_pixel_labels` before it writes a leaf, so a batch
with a label at or above `C` that is not the ignore index stops the step as `Incompatible`
with the pixel named, and the leaves keep the last batch. The check matters beyond the
message: SushiBLAS reads outside the table for an index out of range (`core/graph` README).

`measure` appends `mean_iou` and `pixel_accuracy` through `measure_segmentation`. It picks the
class of each pixel on the host, 4096 pixels at a time, into a `Segment::ConfusionMatrix`
built for the step. That is `R * C` host reads per step: 39 million for one 512 by 512 image
over 150 classes. `TrainerOptions::track_metrics` turns it off. The numbers are those of the
training batch under batch statistics, not of a validation pass. A model of more than 4096
classes trains and fails at its first `measure`.

### pixel_cross_entropy

The loss is the mean, over the pixels that are not ignored, of minus the log-probability of
each pixel's class:

```
log_p   = LOG_SOFTMAX(logits as [R, C])
picked  = GATHER_ROWS(log_p as [R * C, 1], targets)        [R, 1]
loss    = -SUM_ROWS(picked * weights)                      [1]
```

| Leaf | Shape | `fill_pixel_targets` writes |
| --- | --- | --- |
| `targets` | int32 `[R]` | `r * C + mask[r]` for a kept pixel, `r * C` for an ignored one |
| `weights` | real `[R, 1]` | `1 / kept` for a kept pixel, 0 for an ignored one |

The one weight leaf carries both the masking and the divisor. An ignored pixel still gathers a
log-probability, the first of its row, so that its index is valid; the zero weight removes it
from the sum and from the gradient. With the divisor in the leaf the graph needs no scalar
leaf and no `DYNAMIC_SCALE`. Beside the next-token form, the cost is one `[R, 1]` real leaf,
4 bytes a pixel in float32, and one `[R, 1]` `MUL`. `1 / kept` is rounded once to the leaf's
type, a relative error of about 6e-8 in float32. A mask with no kept pixel gives a loss of 0
and a gradient of zeros.

`GATHER_ROWS` takes an int32 index into `R * C` rows, so `N * H * W * C` may not exceed
2 147 483 647. `make` refuses a larger product as `InvalidDescription`. One 512 by 512 image
over 150 classes is 39 321 600 scores, so the largest batch at that size is 54 images; at 1024
by 1024 over 19 classes it is 107.

### dice

The loss is one minus the soft Dice score of the softmax probabilities, averaged over the
classes present, a class being present when a kept pixel holds it:

```
p       = softmax(logits as [R, C])
picked  = GATHER_ROWS(p as [R * C, 1], score_indices)       [R, 1]
I[c]    = SCATTER_ADD_ROWS(picked * kept, class_indices)[c]  p of class c over its own kept pixels
P[c]    = MATMUL(kept transposed, p)[c]                      p of class c over every kept pixel
loss    = SUM_ROWS(class_weights * (1 - 2 I / (P + class_pixels + 1e-6)))
```

| Leaf | Shape | `fill_dice_targets` writes |
| --- | --- | --- |
| `score_indices` | int32 `[R]` | `r * C + mask[r]` for a kept pixel, `r * C` for an ignored one |
| `class_indices` | int32 `[R]` | `mask[r]` for a kept pixel, 0 for an ignored one |
| `kept` | real `[R, 1]` | 1 for a kept pixel, 0 for an ignored one |
| `class_pixels` | real `[C]` | The kept pixels of each class |
| `class_weights` | real `[C]` | `1 / present` for a present class, 0 for an absent one |

The 1e-6 in the denominator is `K_DICE_EPSILON`. It keeps the ratio of an absent class, whose
weight is 0, from being 0 over 0. The loss has no smoothing term in the numerator. A mask with
no kept pixel gives a loss of 0.

Dice is its own objective and not a term of `pixel_cross_entropy`: each of the two has its own
leaves and its own fill. A file trains on both through the `sum` objective of `layers/train`,
which adds the losses of its terms with stated weights and reports `mean_iou` and
`pixel_accuracy` once, from its first term.

An ignored pixel gathers the first probability of its row and names class 0, so both of its
indices are valid; its `kept` of 0 zeroes the gathered value before the scatter-add and removes
it from the gradient. `score_indices` is the index `write_score_indices` also writes for the
cross entropy's `targets`.

No leaf holds `R * C` elements. For one 512 by 512 image over 150 classes in float32, `R` is
262 144: the two index leaves and `kept` are 1 048 576 bytes each and the two class leaves 600
bytes each, so `feed` writes 3 146 928 bytes a step. The gradient crosses the scatter-add
through the `K_SCATTER_ADD_ROWS` rule of `core/autograd`, a row gather by `class_indices`.

`GATHER_ROWS` indexes `R * C` rows in int32, so the bound of the cross entropy holds here too:
`make` refuses `N * H * W * C` above 2 147 483 647 as `InvalidDescription`.
