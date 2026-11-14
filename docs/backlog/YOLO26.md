# Yolo26

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-39 | P2 | open | Add MuSGD, learning rates per parameter group and the weight EMA | MuSGD and the weight EMA exist with tests, and parameter groups carry their own learning rates |  |  | 2026-10-25 |
| RW-40 | P2 | open | Allow one HALF unit below 2^-13 between the reference path and the device | a mixed-precision check that compares the two paths allows one HALF ulp below 2^-13 and bit equality above it (`sushiblas KI-3`) |  |  | 2026-10-19 |
| RW-41 | P2 | open | Cover suppression and the class shift of NMS against the reference | a fixture has a pair above the IoU threshold and more than one class |  |  | 2026-10-19 |
| RW-42 | P2 | open | Close the detect module's open interface points | the owner has ruled on the limits header, and `suppress_overlaps` and the torchvision `nms` are compared with a reference |  | `docs/agent/2026_10_07_YOLO26/PLAN.md` | 2026-10-19 |
| RW-174 | P2 | open | Find why the YOLOv7 one-batch overfit ends at 10.45 percent of its first cost | Large_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth passes with the bound unchanged, or its bound is justified by a Darknet run |  | `modules/vision/darknet_training_cases/source/darknet_training.cpp` | 2026-11-14 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
