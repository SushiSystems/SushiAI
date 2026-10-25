# Ultralytics detectors and YOLOv4-tiny: plan

**Status:** Open — written 2026-10-25. The owner set the scope that day: YOLO26, YOLO11, YOLOv8, YOLOv5 and YOLOv4-tiny train and run from the command line, written without a build; the owner builds once the code of all five is written.

## 1. The goal

Each of the five models has a model file that builds, trains from scratch on coco128 through
`sushiai train` with a loss that falls, and runs through `sushiai inference`, which prints
detections and mAP@0.5. YOLOv4-tiny also imports its stock Darknet weights. The YOLOv3-tiny
tasks T4 to T6 of `docs/agent/2026_10_09_DARKNET_YOLO/PLAN.md` run beside this plan.

The five are examples, not presets. A model is whatever its JSON composes: the library never
asks which generation a model is, and a custom detector is made by editing rows and keys.

Quality comes after a first run: Ultralytics' augmentation, MuSGD and EMA (RW-39), the one-to-one
branch of YOLO26 with top-k, parity with the pinned Ultralytics commit, and Ultralytics' `.pt`
weights (MODEL_IMPORT phase 9, decision D12 of `docs/agent/2026_10_07_YOLO26/SPEC.md`).

## 2. Rulings

| Subject | Ruling, 2026-10-25 |
| --- | --- |
| Models | YOLO26n, YOLO11n, YOLOv8n, YOLOv5n, YOLOv4-tiny |
| No generations | No row, key, value or code path names a model generation. Options describe structure: channels, repeats, kernels, `reg_max`, whether a one-to-one branch exists, the class branch's form |
| Automatic | What the graph already says is derived, never restated: each level's stride from the traced shapes, the head ports from the detect row, `reg_max` and the class count from the head. A key exists only for what the graph cannot say |
| Deterministic | The training file's `deterministic` key, `true` by default: the same files, seed and device give the same weights after every step, bit for bit. Under `true` a key that asks for an unordered path is refused, naming the key. `false` opens unordered paths, such as batches delivered as they finish. The host assignment breaks ties by index; data order comes from the seed |
| YOLOv5 | The original anchor-based YOLOv5 (release 7.0), not `yolov5nu`. It joins the anchor engine of YOLOv3, YOLOv4 and YOLOv7: an `anchor_detection` task, deltas computed on the host, `report_anchor_detections` for its output |
| Anchor-free models | YOLOv8, YOLO11 and YOLO26 share one anchor-free unit |
| Anchor-free loss | A loss graph on the device (D3 of 2026-10-07). The host computes the assignment only and writes it as leaves |
| Builds | None until the code of all five is written. Single files are syntax-checked |
| Mode | Parallel, at most three workers, never idle, no waves. The orchestrator writes the plans and integrates |

## 3. What exists, read on 2026-10-25

- YOLO26's blocks (`conv`, `bottleneck`, `c3k`, `c3k2` with `bottleneck_psa`, `sppf`, `c2psa`,
  `detect`) are registered, `concat` and `upsample` too, and the parser reads `from` and
  `outputs`. `TheYOLO26FileBuildsWhole` builds `applications/examples/yolo26/model.json`.
- `detect` has four direct distances per anchor, no `reg_max`, both branches always, and zero
  biases on its last convolutions.
- `detect/free` holds the anchors, the TaskAlignedAssigner on the host, box coding, and the target
  and anchor leaves. No objective, task section, run row or inference row uses them.
- The anchor engine: the task section `anchor_task.hpp`, `AnchorObjective` (loss
  `-(1/b) sum(delta * head)` over host deltas), `AnchorTrainingStep` with its `HostStage`, the
  run row `anchor` and the inference row `anchor_detection` in `domains_vision.cpp`. Darknet's
  assignment is `head_delta.cpp`: one cell per truth, best anchor by shape plus anchors above a
  threshold. The decode in `slot_decode.hpp` already holds YOLOv5's form under `scale_x_y` 2 and
  the activated size rule.
- YOLOv4-tiny: every key of the stock `.cfg` is read or noted; forward and training parity suites
  exist (`Medium_DarknetYolov4Tiny*`) and have not run with the references set (RW-116).
- Missing: `c2f`, `c3`, DFL, a YOLOv5 assignment and loss, model files for YOLOv8n, YOLO11n and
  YOLOv5n.

## 4. The anchor-free step

The training step splits at the detect ports with the `HostStage` the anchor engine uses. The
forward runs to the ports; the host decodes the one-to-many ports, runs the TaskAlignedAssigner
and writes the target leaves; the rest of the step evaluates the loss graph over the ports and
the leaves, differentiates it and steps the optimiser. One forward per step, where D2 of
2026-10-07 planned two.

The loss graph follows Ultralytics' detection loss: binary cross-entropy of the class logits
against the target scores, `1 - CIoU` of the decoded boxes weighted by the target scores, and,
when `reg_max` is above 1, the distribution focal loss. The gains default to 0.5, 7.5 and 1.5 and
are keys of the training file. Each term is divided by the sum of target scores. The CIoU term
guards its divisions with `K_SELECT` as section 4 of the YOLO26 `SPEC.md` says.

The task section holds what the graph cannot say: the score and suppression thresholds and the
assigner's settings. The objective reads strides, ports, `reg_max` and classes from the built
model.

## 5. Tasks

Hierarchy first: blocks, then task sections and model files, then objectives, then the program
rows, then example folders. Each task has one acceptance criterion. "Written" means files
syntax-checked one by one and tests the owner's build runs.

| Id | Task | Waits on | Accepted when |
| --- | --- | --- | --- |
| U1 | `c2f` and `c3` rows in `vision/blocks` | nothing | rows registered; tests pin state names, shapes and one forward value |
| U2 | `detect` options: `reg_max`, `end2end`, `class_branch`, and the bias initialisation from strides derived from the trace | nothing | YOLO26's 594 records unchanged; tests pin a head with `reg_max` 16 and no one-to-one branch, and its biases |
| U3 | DFL: the expectation of a `4 * reg_max` box output, as graph and on the host, and its loss term | nothing | graph and host agree with a hand-worked case; gradcheck passes on the loss term |
| U4 | The anchor-free task section and the model facts derived from the build: ports, strides, `reg_max`, classes | U2 | a written section reads back field by field; the facts of the YOLO26 file equal hand-worked values |
| U5 | Model files `yolov8n`, `yolo11n`, `yolov5n` under `applications/examples/`, and the task section in `yolo26` | U1, U2, U4 | a registration test builds each and pins its parameter count against Ultralytics' summary |
| U6 | The anchor-free assignment stage: decode the ports, assign, fill the target leaves | U3, U4 | the leaves of a two-image fixture equal hand-worked targets |
| U7 | The anchor-free loss graph | U3, U6 | its value equals a host evaluation on the U6 fixture, and gradcheck passes |
| U8 | The anchor-free objective, training step and run row | U7 | a program test trains a two-level fixture model for two steps, twice, with equal weights |
| U9 | The anchor-free inference row: decode, NMS, print, mAP | U4, T4 | a test prints the detections and mAP of a hand-worked fixture |
| U10 | YOLOv5's assignment in the anchor engine: the anchor ratio gate and neighbour cells, behind the interface Darknet's assignment implements | nothing | tests pin the cells and anchors of hand-worked truths |
| U11 | YOLOv5's loss in the anchor engine: objectness balance per level, IoU objectness targets, class and CIoU terms, gains scaled by levels and classes | U10 | host deltas equal a central difference of the host loss on a fixture |
| U12 | The anchor head's bias initialisation and the `yolov5n` task section | U10 | the `yolov5n` model file builds with its anchor task; biases pinned |
| U13 | Example folders for the five models: `train.json` and `eval.json` on coco128, a README; YOLOv4-tiny's from `sa import` | U5, U8, U9, U12 | each README names the commands the owner's runs use |
| U15 | The `deterministic` key: read and written with the training file, `true` by default, refusing every unordered path under `true`; the loader and every random draw audited for order | nothing | tests pin the default, the refusal naming the key, and two seeded loader epochs delivering equal batches |
| U14 | Manual and backlog: design topics, `RUNNING_MODELS.md`, `MODEL_FILE.md`, changelog | each task | `check_docs_truth.py` passes; each task's changelog line is in |

The owner's build follows U14, then the runs for each model: one batch overfits, ten epochs on
coco128 lower the loss, two runs with the same seed end with equal weights, and inference prints
detections and mAP.

## 6. Not verified

- That ten epochs from scratch on 128 images give boxes worth printing. A run from scratch on so
  few images may only memorise them.
- Ultralytics' loss gains, bias formulas and parameter counts are quoted from memory of
  Ultralytics' source; U5, U11 and U12 check them against the source when written.
- That a coco128 batch of 16 at 640 pixels fits the memory plan.
- That every SushiBLAS kernel a step uses reduces in a fixed order; the determinism run of the
  owner's build is the first check.
