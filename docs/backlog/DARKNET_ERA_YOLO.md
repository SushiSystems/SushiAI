# Darknet era yolo

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-46 | P2 | unmeasured | Confirm weight-only decay against Darknet | a parity suite states a decay and matches Darknet, which decays convolution weights and neither biases nor batch norm scales |  |  | 2026-10-19 |
| RW-50 | P2 | open | Delete the temporary Darknet checkout | the checkout outside the repository is gone, once the training tests pass |  |  | 2026-10-19 |
| RW-192 | P2 | open | Find why the unpacked step of YOLOv3-tiny returns no tensors under load | Medium_DarknetYolov3TinyTraining.ThePlanOfTheStepChangesNoValue passes in 20 gating runs in a row, with its cause named |  | `modules/vision/darknet_training_cases/source/darknet_training.cpp` | 2026-11-24 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-23 | own | ThePlanOfTheStepChangesNoValue fails under parallel load, unpacked has 0 tensors | Rerun the shard alone; the runner already does | RW-192 | `modules/vision/darknet_training_cases/source/darknet_training.cpp` | 2026-11-24 |
