# tests/vision/importers/test_ultralytics_model.cpp

Suites: `Medium_UltralyticsHandFile`, `Small_UltralyticsModel`.

The descriptions are JSON spellings of the `yaml` dictionary an Ultralytics checkpoint pickles.
The YOLO26n one is `ultralytics/cfg/models/26/yolo26.yaml` at the pinned commit with the scale "n".

## TheDescriptionIsTheHandWrittenModelFile

Suite `Medium_UltralyticsHandFile`, one row per released checkpoint: the `yaml` of `yolo26n.pt`,
`yolov8n.pt` and `yolo11n.pt`, translated at side 640, writes the same model file as
`model.json` of `applications/examples/yolo26`, `yolov8n` and `yolo11n`. Each row passes the class
its checkpoint pickles at `9.cv1.act`: Identity for YOLO26n, SiLU for the other two.

## AnAnchoredDetectBecomesOneConvolutionPerLevel

A YOLOv5 Detect over rows 0 and 1, two anchors a level and two classes, becomes a 1x1
convolution per level of 2 x (2 + 5) = 14 filters; the second reads row 1.

## AnAnchorIsAFractionOfTheInputSide

The anchor 10 x 13 pixels at side 64 is stated as 10 / 64 and 13 / 64.

## TheTensorsOfALevelTakeTheRowOfItsConvolution

The tensors of `2.m.1.` are renamed to `3.`, the row of the second level's convolution.

## AStatedPaddingIsKeptOnTheConvRow

The stem `Conv [64, 6, 2, 2]` keeps padding 2 on its conv row.

## TheMScaleTurnsTheInnerBlockOfC3k2IntoC3k

A C3k2 entry that states `c3k` false, in a description of the scale "m", names the inner block
"c3k", as Ultralytics' own model builder does for the m, l and x scales.

## AWidthIsCappedScaledAndRoundedUpToEight

A Conv of 1000 channels under a cap of 512 and a width of 0.3 writes 160 filters: 512 times 0.3
is 153.6, which rounds up to the next multiple of eight.

## EveryUnreadModuleIsListedWithItsEntry

A description with `Focus` at entry 1 and `RepNCSPELAN4` at entry 2 fails with `Unsupported`, and the
message names both with their entries.

## ADescriptionWithoutADetectEntryIsRefused

A description with no Detect entry fails with `Unsupported`, "no Detect entry".

## APickledSiluAfterTheSppfNarrowingIsKept

An SPPF at entry 1 whose checkpoint pickles `SiLU` at `1.cv1.act` writes `activate_narrowing` true.

## AnSppfWithoutAPickledClassKeepsItsNarrowingLinear

The same SPPF with no pickled classes writes no `activate_narrowing`, so its cv1 stays linear.
