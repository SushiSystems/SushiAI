# Yolo26

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-35 | P2 | open | Build the detection objective | code builds the loss graph of section 4 of the YOLO26 specification |  | `docs/agent/2026_10_07_YOLO26/PLAN.md` | 2026-10-19 |
| RW-36 | P2 | open | Assign labels in two passes in the training loop | an anchor-free training step replays forward only, assigns with `detect/free`, and runs the full step in the loop |  |  | 2026-10-19 |
| RW-39 | P2 | open | Add MuSGD, learning rates per parameter group and the weight EMA | MuSGD and the weight EMA exist with tests, and parameter groups carry their own learning rates |  |  | 2026-10-25 |
| RW-40 | P2 | open | Allow one HALF unit below 2^-13 between the reference path and the device | a mixed-precision check that compares the two paths allows one HALF ulp below 2^-13 and bit equality above it (`sushiblas KI-3`) |  |  | 2026-10-19 |
| RW-41 | P2 | open | Cover suppression and the class shift of NMS against the reference | a fixture has a pair above the IoU threshold and more than one class |  |  | 2026-10-19 |
| RW-42 | P2 | open | Close the detect module's open interface points | the owner has ruled on the limits header, and `suppress_overlaps` and the torchvision `nms` are compared with a reference |  | `docs/agent/2026_10_07_YOLO26/PLAN.md` | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
