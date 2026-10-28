# Model import

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-51 | P2 | unmeasured | Run the writers and the task section | `write_json`, `write_model`, `AnchorTask` and their tests pass in a recorded run |  | `docs/design/MODEL_IMPORT/README.md` | 2026-10-25 |
| RW-54 | P2 | open | Let the `idx` reader row name its two files, and a row set the letterbox | both rows exist |  |  | 2026-10-19 |
| RW-55 | P2 | open | Add subdivisions and options per head | the training file takes `subdivisions`, and each head of the anchor objective takes its own options |  | `docs/design/MODEL_IMPORT/README.md` | 2026-10-25 |
| RW-56 | P1 | unmeasured | Load the files the import writes | the three files the import writes for each of the six models equal their expected text, and `load_checkpoint` accepts each written `weights.sushi` |  |  | 2026-10-25 |
| RW-57 | P2 | open | Set detection thresholds in `eval.json` and infer on one image file | both work |  |  | 2026-10-19 |
| RW-58 | P2 | open | Read the imported files in the parity suites | the forward and training parity suites load the `model.json` and `weights.sushi` the import wrote, not the `.cfg` |  |  | 2026-10-25 |
| RW-132 | P1 | unmeasured | Run both ends of the anchor path in a program test | a test under `tests/applications/programs/` trains an anchor model over a small YOLO-layout fixture through the train program's anchor row, and runs… |  |  | 2026-10-25 |
| RW-133 | P2 | unmeasured | Give YOLOv3-tiny an example directory | `applications/examples/yolov3_tiny/` holds a `train.json` with a filled `yolo_txt` data block and a README a run can follow |  |  | 2026-10-25 |
| RW-135 | P2 | unmeasured | Report mean average precision in `sushiai inference` | inference prints mAP at an IoU of 0.5 against the labels of the evaluated split |  |  | 2026-10-25 |
| RW-136 | P2 | open | Carry the `.cfg`'s augmentation into the training file | `jitter`, `hue`, `saturation` and `exposure` are training-file keys the import writes |  |  | 2026-10-25 |
| RW-137 | P2 | open | Write a float32 setting as the shortest decimal that reads back | the import writes the `.cfg`'s `ignore_thresh=.7` as `0.7`, and a written float32 value parses back to the same float32 |  |  | 2026-10-25 |
| RW-138 | P2 | open | Letterbox an imported `letter_box=1` model as Darknet does | an image fitted to a model imported with `letter_box=1` is padded with 0.5 and may be scaled up, as Darknet's `letterbox_image` does, and a test pins one such… |  |  | 2026-10-25 |
| RW-139 | P2 | open | Give the tests one scratch directory | the private `ScratchDirectory` classes of `test_imported_model_writer.cpp`, `language/datasets/datasets_test_support.hpp` and… |  |  | 2026-10-25 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
