# Darknet era yolo

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-45 | P2 | open | Follow the first SushiBLAS tag | CI pins `SUSHIRUNTIME_REF` and `SUSHIBLAS_REF` to tags |  | `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md` | 2026-10-25 |
| RW-46 | P2 | unmeasured | Confirm weight-only decay against Darknet | a parity suite states a decay and matches Darknet, which decays convolution weights and neither biases nor batch norm scales |  |  | 2026-10-19 |
| RW-48 | P1 | open | Train YOLOv3-tiny from the command line | `sushiai import darknet` of `yolov3-tiny.cfg`, `sushiai train` and `sushiai inference` have run end to end on a small YOLO-layout dataset and printed detections |  |  | 2026-10-25 |
| RW-49 | P2 | open | Map boxes back through the letterbox | a function maps boxes from the letterboxed input to the source image, and `sushiai inference` prints them in source pixels |  |  | 2026-10-25 |
| RW-50 | P2 | open | Delete the temporary Darknet checkout | the checkout outside the repository is gone, once the training tests pass |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
