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
