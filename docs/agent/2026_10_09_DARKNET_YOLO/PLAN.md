# Darknet-era YOLO: plan of the YOLOv3-tiny milestone

**Status:** Open — rewritten 2026-10-25 for the milestone the owner chose that day: YOLOv3-tiny imported, run and trained from the command line. Requirements are in `SPEC.md` beside this file; Darknet's behaviour is in `REPORT.md`.

## 1. The milestone

`sushiai import darknet` turns `yolov3-tiny.cfg` and `yolov3-tiny.weights` into a model file, a
weight file and a training file. `sushiai inference` runs the imported weights over coco128 and
prints detections and a mean average precision that sits near Darknet's on the same images.
`sushiai train` then trains from those weights on coco128, the loss falls, and the trained
checkpoint runs through `sushiai inference` again. One training step is timed against a Darknet
iteration on the same card.

Every link of that chain is written (`domains_vision.cpp`, `import.cpp`, the anchor objective, the
`yolo_txt` reader). None has run end to end, and the parity suites that guard it were skipped in
the last test run because `SA_REFERENCE_DATA_DIR` was unset.

## 2. Rulings and inputs

| Subject | Value |
| --- | --- |
| Model | `yolov3-tiny`, Darknet's stock `.cfg` and `.weights` |
| References | `SA_REFERENCE_DATA_DIR=D:/Projects/.sushisystems/models`; the Darknet build beside them computes the reference mAP and step time |
| Data | coco128 at `D:/Projects/.sushisystems/datasets/coco128`, split `train2017`, 128 images, outside the repository |
| Image fit | The model follows its `.cfg`: stretch under `letter_box=0`, letterbox under 1 (RW-134, ruled 2026-10-25) |
| Mode | Inline, one session, tasks in the order of section 3 (ruled 2026-10-25) |
| Builds | The owner's machine, through `sa`, at the build points of section 4, with `CMAKE_BUILD_PARALLEL_LEVEL=6` |
| Memory | No run raises `--memory-limit`. Batch size starts at 16 and grows only on a measured plan |

Out of scope, as `SPEC.md` rules: Darknet's data loader and augmentation (RW-136 stays in the
backlog), a full COCO run, the larger models.

## 3. Tasks

Hierarchy first: the runs that need no code come before the code, and the file format changes
before the drivers that read them. Each task has one acceptance criterion.

| Id | Task | Backlog | Waits on | Accepted when |
| --- | --- | --- | --- | --- |
| T0 | Run the YOLOv3-tiny parity scenarios and the import scenarios with the references set | RW-116 | nothing | `REPORT.md` records each `Medium_DarknetYolov3Tiny*`, `Models/Small_DarknetDecode` and `ParityModels/Medium_DarknetImportEndToEnd` case as passed or failed, with the failing lines |
| T1 | Import YOLOv3-tiny and run the imported weights over coco128 | RW-48, first half | T0 | `sushiai inference` prints detections for coco128 with one compile; the classes of the first images are recorded |
| T2 | Fit images as the `.cfg` says | RW-134 | T1 | an imported `letter_box=0` model trains and infers on stretched images, a `letter_box=1` model imports and letterboxes, with a test for each |
| T3 | Load the files the import writes | RW-56 | T2 | for each of the six models the written `model.json`, `train.json` and `weights.sushi` parse, and `load_checkpoint` accepts the weights |
| T4 | Report mean average precision in inference | RW-135 | T2 | `sushiai inference` prints mAP at IoU 0.5 over the labelled split, and a test pins it on a fixture whose answer is computed by hand |
| T5 | Test both ends of the anchor path in the programs | RW-132 | T2 | a program test trains an anchor model for two steps over a small YOLO-layout fixture and runs `report_anchor_detections` on its checkpoint |
| T6 | Give YOLOv3-tiny an example directory | RW-133 | T2 | `applications/examples/yolov3_tiny/` holds `train.json`, `eval.json` and a README; the README's commands are the ones T7 ran |
| T7 | Compare pretrained mAP with Darknet on coco128 | RW-48 | T4, T6 | `REPORT.md` holds SushiAI's and Darknet's mAP@0.5 of the stock weights on coco128 `train2017`, and the owner has ruled whether the difference holds |
| T8 | Train YOLOv3-tiny on coco128 from the stock weights | RW-48 | T7 | a run of 10 epochs writes a checkpoint, its loss at the last epoch is below the first, and inference of that checkpoint prints detections and its mAP |
| T9 | Time a training step against Darknet | RW-102 | T8 | `REPORT.md` holds the median step time of both at one batch size on the RTX 3080 Ti, with the SushiBLAS commit |

RW-49 (boxes in source pixels) and RW-57 (thresholds in `eval.json`) join T4 if the mAP code needs
boxes in source pixels; otherwise they stay in the backlog.

## 4. Build points

| Point | After | What the owner runs |
| --- | --- | --- |
| B0 | nothing | `sa build`, then `sa test -l medium -f "DarknetYolov3Tiny\|DarknetDecode\|DarknetImportEndToEnd"` with `SA_REFERENCE_DATA_DIR` set (T0); T1 is a run of `sa import` and `sa inference` |
| B1 | T2, T3 | `sa build`, the gating tests and the T0 filter again |
| B2 | T4, T5, T6 | `sa build`, the gating tests, then the runs of T7 |
| B3 | T7 | the runs of T8 and T9; no build unless they find a defect |

## 5. Where the fit lives

The owner ruled on 2026-10-25 that the fit lives in the task section of the model file. The import
writes `letter_box`, and the vision driver reads it and hands the fit to the dataset. The fit
travels with the weights that were trained under it, so an `eval.json` written by hand cannot
disagree with it. A data block that names its own fit is refused when it differs from the model's.

## 6. Not verified

- That the imported files load: T3 is the first test that reads them from disk.
- That a coco128 batch of 16 at 416 by 416 fits the memory plan of a training step.
- That `report_anchor_detections` prints sensible boxes: no test runs its body today.
