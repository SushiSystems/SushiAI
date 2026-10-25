# Ultralytics detectors and YOLOv4-tiny: report

**Status:** Open — opened 2026-10-25 with the plan beside this file.

## 1. Dispatches

dispatch: map the anchor-free pieces — haiku medium, 29 turns, passed, rung 1
dispatch: map YOLOv5 and YOLOv4-tiny — haiku medium, 23 turns, passed, rung 1
dispatch: map the training and inference rows — haiku medium, 53 turns, passed, rung 1
dispatch: check that the YOLO26 model file builds — haiku medium, 26 turns, passed, rung 1
dispatch: U1 c2f and c3 — sonnet medium, 49 turns, passed, rung 1
dispatch: U2 detect options — sonnet medium, 71 turns, passed, rung 1
dispatch: U10 shape-ratio assignment — opus low, 43 turns, passed, rung 1
dispatch: U15 deterministic key — sonnet medium, 44 turns, passed, rung 1
dispatch: U3 distribution focal loss — opus low, 58 turns, passed, rung 1
dispatch: U4 task section and facts — opus low, 76 turns, passed, rung 1
dispatch: U11 shape-ratio loss — opus low, 42 turns, passed, rung 1
dispatch: U12 yolov5n file and anchor biases — sonnet medium, 77 turns, passed, rung 1
dispatch: U6 anchor-free assignment stage — opus low, 41 turns, passed, rung 1
dispatch: conv padding and RW-142 — sonnet medium, 45 turns, passed, rung 1
dispatch: wire the shape-ratio rule — opus low, 112 turns, passed, rung 1
dispatch: U5 model files — sonnet medium, 44 turns, passed, rung 1
dispatch: U7 anchor-free loss graph — opus low, 76 turns, passed, rung 1

## 2. Seen while mapping

- The anchor objective is built twice on the run path: `TrainingSetup::make` builds one through
  the registry and `run_anchor` builds another (`training_setup.cpp:120`,
  `domains_vision.cpp:92`).
- `TheYOLO26FileBuildsWhole` reads a manifest from a fixture folder git ignores; whether it skips
  or fails on a fresh clone was not checked.

## 3. U1, written and not built

`C2f` is `C3k2<Bottleneck>` built with bottlenecks as wide as the hidden width; `C3` is `C3k`
through a new constructor that takes both bottleneck kernels, 1 and 3. Both rows read `filters`,
`hidden`, `repeats`, `shortcut` and the norm keys, as `c3k2` does. The block tests check the
wiring of a traced `[1, 2, 2, 8]` input, not output values: the block tests have no engine. Each
block test file repeats the same helpers.

## 4. U2, written and not built

`DetectOptions` holds `reg_max`, `end2end`, `class_branch` and `strides`; the branches are a
`std::variant` of the two class-branch forms. Defaults keep the YOLO26 file's 594 records. A
level's stride cannot be derived inside the row: its factory sees only the per-level shapes. The
row therefore takes no `strides` key, and U4 derives the strides from the built model and
applies the bias initialisation on a fresh run. Ultralytics' `dfl.conv.weight`, a fixed ramp, has
no state record; a later `.pt` mapping skips it. The state names follow Ultralytics' as
remembered, not checked against a checkpoint.

## 5. U10, written and not built

`AssignmentRule` (`target_assignment.hpp`) turns the truths of a batch into `TargetAssignment`
records: image, anchor slot, cell, the box relative to the cell, label and truth row.
`ShapeRatioAssignment` keeps an anchor whose width and height ratios to the truth stay under
`ratio_limit` (4 by default) and adds the neighbour cells within `neighbour_offset` (0.5) of the
centre. The anchor task's `assignment` member chooses `best_shape`, the default and Darknet's
rule, or `shape_ratio`; a section without it reads and writes as before.

Two deviations from Ultralytics, both for determinism: when two truths claim one cell and anchor,
the higher shape IoU wins and the lower truth row breaks a tie, where Ultralytics leaves it to
scatter order; and `neighbour_offset` is limited to [0, 0.5].

Darknet's `head_delta.cpp` assigns and writes deltas in one pass and does not go through the new
seam, so `best_shape` has no `AssignmentRule` type yet. Splitting it waits for a build that runs
the Darknet parity suites (RW-140).

## 6. U15, written and not built

`TrainSpec::deterministic` is read from training file version 3, true when absent; versions 1 and
2 refuse the key as unknown, and the import writes version 3. A new loader case checks that one
worker and four workers with another prefetch deliver the same two epochs.

The order audit found every path ordered: the loader hands batch `index` by slot whatever its
workers; sample order is `EpochPermutation(seed, epoch, count)`; augmentation, dropout, replica
draws and initialisation are counter-based or seeded once; the `unordered_map` loops in
`eager/session.cpp` only set flags or collect tensors to zero; clocks only time. The region
executors were searched for the patterns, not read line by line.

One unordered path sits behind a key: a cluster replica's `if_absent: "local"` runs a lost
worker's replica on the master's device. Under `deterministic: true` it is to be refused
(RW-141).

## 7. U3, written and not built

Bin `b` of side `s` sits at channel `s * reg_max + b`, sides left, top, right, bottom, as
Ultralytics' `view(b, 4, reg_max, a)` reads it; taken from memory of Ultralytics, not its source.
The host forms are in `detect/free`, the graph forms in `vision/training`, since `detect` does not
depend on the graph module. The expectation reshapes to `[N * 4, reg_max]`, takes a softmax and
multiplies by a `[reg_max, 1]` leaf of the bin values, because `K_CONSTANT` holds one value; at
`reg_max` 1 it emits nothing. The loss is `-sum(log_softmax * weights)` over host-built weights:
`wl` at the lower bin and `wr` at the upper, each times the score weight over the score sum and a
quarter for the mean over sides, so the mask, the normalisation and the mean live in the weights.
The worker did not load `cpp-code-style` or `testing`; the files pass the comment checker and the
syntax check.

## 8. U4, written and not built

The `anchor_free_detection` section holds `score_threshold`, `suppression` and an optional
`assignment` with a `one_to_many` and a `one_to_one` branch over `AssignerConfig`.
`derive_free_model_facts()` walks the published `one2many.level<l>` ports and returns, per level,
the size, the stride and the port names of each copy, with `reg_max`, the class count and whether
a one-to-one copy exists; a stride that does not divide the input is refused naming the port.
`initialize_free_biases()` runs after `NN::initialize` and writes the last biases by state name,
all entries checked before any is written. A setter on `DetectHead` would come too late: a
convolution reads its bias start on the first trace, and the strides are known only after it.

Both files repeat the head's port and branch names and the constants 1, 5 and 640;
`detect_head.hpp` should publish them (RW-142). The facts and the bias code live in
`vision/training`, since `detect` depends on neither the trace nor the model state.

## 9. U11, written and not built

The head is `[B, H, W, A * (5 + C)]`, each slot x, y, w, h, objectness, classes: Ultralytics'
`[b, na, h, w, 5 + nc]` with the anchors moved last. Only logit coding is accepted. The delta is
`-B` times the gradient of the total loss, which already carries Ultralytics' factor of the batch.
Objectness targets are held fixed. The worker loaded `sushiskills` and `humanizer` only; the files
pass the comment checker and the syntax check.

Left for the next dispatch: the training step does not call the unit yet, since the anchor
objective carries no assignment choice and its cost is Darknet's; and the box gradient
differentiates CIoU's `alpha`, which Ultralytics holds constant.

## 10. U6, written and not built

`FreeAssignmentStage` joins the one-to-many ports of every level, decodes the distances (the
distribution expectation when `reg_max` is above 1), takes the sigmoid of the scores, runs the
TaskAlignedAssigner with the task's settings and fills ten leaves per level, roles
`free.<kind>.level<l>`, plus `free.target_scores_sum`. Top-k ties go to the lower index and a
contested anchor to the first truth of the largest overlap, as the assigner already does. One case
is pinned on purpose: anchors inside a truth whose alignment metric is 0 can win the top-k by
index and become foreground with a target score of 0, as Ultralytics' assigner allows. Three
switches over the closed `FreeLeafKind` name, size and source the leaves.

## 11. The conv padding and the head contract

The `conv` row takes `padding`, half the kernel when absent, so YOLOv5's stem is one row of
kernel 6, stride 2 and padding 2. The worker published the head's names from `detect_head.hpp`,
which made `vision/training` depend on `vision/blocks`. The owner asked for the edge only if it
kept SOLID; it does not, since the training code would depend on a concrete block for what is a
contract between the head and the loss. The names and the class bias formula moved to
`vision/detect` (`free/head_contract.hpp`), which `blocks` now depends on and `training` already
did; `detect_head.hpp` re-exports them with using-declarations, and the formula's two copies are
one.

## 12. The shape-ratio rule in the step, written and not built

`HeadDeltaRule` computes a batch's deltas and its cost in one call, so the shape-ratio assignment
runs once per batch. `choose_head_delta_rule` is the one switch over `AssignmentChoice`, run when
the objective is built; both builds on the run path go through it. `AnchorObjective::measure_cost`
now serves only the Darknet rule.

The box gradient of the shape-ratio rule holds CIoU's trade-off weight constant, as Ultralytics
computes it under `no_grad` (`BoxGradient::FixedTradeOff`); its difference test freezes the weight
at the unshifted head. The epsilon is 1e-6 where Ultralytics uses 1e-7.

`yolov5n`'s heads are the three convolutions' logits, with `"size": "squared_sigmoid"` decoding
`(2 * sigmoid)^2` times the anchor. The Darknet forward parity code reads no `size`.

## 13. U5, written and not built

`yolov8n` has 23 rows and `yolo11n` 24; with `yolo26` they carry an `anchor_free_detection`
section at the defaults. A count over the rows gives 3,157,184 parameters for YOLOv8n and
2,624,064 for YOLO11n, 16 under Ultralytics' summaries as remembered (3,157,200 and 2,624,080):
the 16 values of the distribution ramp, which has no state record here. The test pins the
summary minus 16. The C3k2 rows of `yolo11n` take `bottleneck` or `c3k` as the yaml states for
scale n. Three registration test files now each hold a copy of the same composing helper.

## 14. U7, written and not built

Per level, over the one-to-many ports: the class term is `softplus(x) - x * t` summed; the box
term decodes the distances (through the distribution expectation), swaps background anchors for
`safe_boxes` with a `SELECT` on `box_mask`, and sums `(1 - CIoU) * normalized_weight` with CIoU's
trade-off weight behind `STOP_GRADIENT` and Ultralytics' epsilon of 1e-7; the distribution focal
term runs when `reg_max` is above 1; an L1 term on the distances runs when `l1_gain` is above 0,
which it is not by default. Only the class term is divided by the score sum: the box, focal and
L1 leaves already carry that division. Each term is scaled by its gain times the batch, so the
loss root needs no seed factor. `anchor_strides` goes unused, the loss working in grid units, and
the objective must declare the `[reg_max, 1]` bin-value leaf the expectation multiplies by.

The gradcheck covers box, class and L1 together where the predicted and target boxes share an
aspect ratio, since a central difference cannot see a weight held constant elsewhere.
