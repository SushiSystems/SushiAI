# Darknet era yolo

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-46 | P2 | unmeasured | Confirm weight-only decay against Darknet | a parity suite states a decay and matches Darknet, which decays convolution weights and neither biases nor batch norm scales |  |  | 2026-10-19 |
| RW-48 | P1 | open | Train YOLOv3-tiny from the command line | `sushiai import darknet` of `yolov3-tiny.cfg`, `sushiai train` and `sushiai inference` have run end to end on a small YOLO-layout dataset and printed detections |  |  | 2026-10-25 |
| RW-49 | P2 | open | Map boxes back through the letterbox | a function maps boxes from the letterboxed input to the source image, and `sushiai inference` prints them in source pixels |  |  | 2026-10-25 |
| RW-50 | P2 | open | Delete the temporary Darknet checkout | the checkout outside the repository is gone, once the training tests pass |  |  | 2026-10-19 |
| RW-163 | P2 | decision | Resolve the yolov4-tiny second mask the anchor head assignment gives | assign_anchors_to_heads and the cfg agree on the second mask, or the departure is documented, and a derived test replaces the characterization |  | `tests/vision/detect/anchor/test_head_assignment.cpp` | 2026-11-09 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
