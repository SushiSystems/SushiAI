# Ultralytics detectors and YOLOv4-tiny: report

**Status:** Open — opened 2026-10-25 with the plan beside this file.

## 1. Dispatches

dispatch: map the anchor-free pieces — haiku medium, 29 turns, passed, rung 1
dispatch: map YOLOv5 and YOLOv4-tiny — haiku medium, 23 turns, passed, rung 1
dispatch: map the training and inference rows — haiku medium, 53 turns, passed, rung 1
dispatch: check that the YOLO26 model file builds — haiku medium, 26 turns, passed, rung 1
dispatch: U1 c2f and c3 — sonnet medium, 49 turns, passed, rung 1

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
