# tests/applications/programs/source/test_anchor_run.cpp

Suites: `Medium_AnchorRun`, in `sushiai_programs_tests`.

Runs the `anchor` row of the run table and the `anchor_detection` row of the report table over
a fixture, each through the steps the program takes: `TrainingSetup::make`, `run_training`,
`save_run_checkpoint`, then `load_checkpoint`, `EvaluationStep::make` and `report_inference`. The
two helpers in the file repeat the assembly of `train.cpp` and `inference.cpp`, which hold it in
their `main`.

The fixture is `tests/applications/programs/fixtures/tiny_yolo/`: a model file of two
convolutions with one head of two anchors and two classes over a 16 by 16 input, and a
YOLO-layout dataset of four 16 by 16 PNG images with one or two labelled boxes each, under
`images/train` and `labels/train`. The tests read it in place through `SA_TEST_FIXTURE_DIR` and
write only the checkpoint, under the temporary directory.

## TheAnchorRowTrainsTwoStepsAndWritesACheckpoint

One epoch of two batches of two images is two updates. The optimizer has taken 2 steps, the run
reports one compile, and the checkpoint file exists.

## TheAnchorReportReadsTheCheckpointOfTheRun

A run is trained as above, and the report loads its checkpoint. The report names one head and two
classes, ends the detections with "detections over 4 images.", and prints the mean average
precision line over 4 images, which only a split with a `boxes` role reaches. The test states
no detection count and no precision: two updates of an untuned model fix neither.
