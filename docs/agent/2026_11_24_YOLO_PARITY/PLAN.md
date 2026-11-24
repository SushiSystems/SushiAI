# YOLO parity work without a run

**Status:** Open — written, not built or run; the orchestrator runs the `sa` commands below.

No build, test or speed run is allowed in this task. Each item gets what can be written, then
the `sa` command the orchestrator must run.

1. RW-116: read the scenario cases, list the run command.
2. RW-46: read the training parity cases and Darknet's update rule; write a decay case if the
   Darknet reference can be read from source.
3. RW-104: read `test_yolov3.cpp`; find what per-layer dump exists.
4. RW-101: same for `test_yolov4.cpp` and `test_yolov7.cpp`.
5. RW-102: list the measurement command; no figure.
6. RW-103: list the run command; no figure.
7. RW-74: read `tools/reference/export_yolo26_reference.py` and the version 2 reader; write it.
8. RW-133: read `applications/examples/yolov3_tiny/`; fill `train.json` and the README.
9. List other open YOLO, Darknet and Ultralytics items.
