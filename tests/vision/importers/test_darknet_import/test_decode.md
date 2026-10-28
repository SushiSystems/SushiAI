# tests/vision/importers/test_darknet_import/test_decode.cpp


Suites: `Small_DarknetDecode`, in `sushiai_vision_tests`.

The suite is parameterised over six models: `yolov3-tiny`, `yolov7-tiny`, `yolov7`, `yolov3`,
`yolov4-tiny` and `yolov4`. For each it reads the `.cfg` from the directory `SA_REFERENCE_DATA_DIR`
names and, for every [yolo] head, three reference tensors under `references/<run>/`: the raw head
input, the decoded boxes and the decoded scores before any threshold. The weights are not read. The
decode runs on the host through `decode_head`.

## TheHeadsDecodeToTheDumpedRows

The translated model has the head count the file lists, each head at the listed layer with 80
classes and options that `validate_head` accepts. The reference input has the extents `[grid, grid,
channels]`, the boxes `[rows, 4]` and the scores `[rows, 80]`. Every decoded box entry and score is
within 1e-6 absolute of the reference, and the largest difference is printed. The case is skipped,
naming the variable or the first missing file, when the reference data is absent.
