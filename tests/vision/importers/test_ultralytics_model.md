# tests/vision/importers/test_ultralytics_model.cpp

Suites: `Medium_UltralyticsModel`, `Small_UltralyticsModel`.

The descriptions are JSON spellings of the `yaml` dictionary an Ultralytics checkpoint pickles.
The YOLO26n one is `ultralytics/cfg/models/26/yolo26.yaml` at the pinned commit with the scale "n".

## TheYolo26nDescriptionIsTheHandWrittenModelFile

`Medium_`, since it reads `applications/examples/yolo26/model.json`. The YOLO26n description
translated at a side of 640 writes the same model text as that file, which was written by hand
from the paper and the Ultralytics source.

## TheMScaleTurnsTheInnerBlockOfC3k2IntoC3k

A C3k2 entry that states `c3k` false, in a description of the scale "m", names the inner block
"c3k", as Ultralytics' own model builder does for the m, l and x scales.

## AWidthIsCappedScaledAndRoundedUpToEight

A Conv of 1000 channels under a cap of 512 and a width of 0.3 writes 160 filters: 512 times 0.3
is 153.6, which rounds up to the next multiple of eight.

## EveryUnreadModuleIsListedWithItsEntry

A description with `Focus` at entry 1 and `C2f` at entry 2 fails with `Unsupported`, and the
message names both with their entries.

## ADescriptionWithoutADetectEntryIsRefused

A description with no Detect entry fails with `Unsupported`, "no Detect entry".
