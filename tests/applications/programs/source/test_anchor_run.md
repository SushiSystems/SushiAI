# tests/applications/programs/source/test_anchor_run.cpp

Suites: `Medium_AnchorRun`, in `sushiai_programs_tests`.

Runs the `anchor` row of the run table and the `anchor_detection` row of the report table over
a fixture, each through the function the program calls: `run_training_file` of
`training_program.hpp` and `run_inference_file` of `inference_program.hpp`. `train.cpp` and
`inference.cpp` keep only their arguments and their exit code, so a change to the assembly
reaches this test.

The fixture is `tests/applications/programs/fixtures/tiny_yolo/`: a model file of two
convolutions with one head of two anchors and two classes over a 16 by 16 input, and a
YOLO-layout dataset of four 16 by 16 PNG images with one or two labelled boxes each, under
`images/train` and `labels/train`. The tests read it in place through `SA_TEST_FIXTURE_DIR` and
write only the checkpoint, under the temporary directory.

## TheAnchorRowTrainsTwoStepsAndWritesACheckpoint

One epoch of two batches of two images is two updates. The optimizer has taken 2 steps, the run
reports one compile, the checkpoint file exists, and neither the header nor the checkpoint line
counts zero learnable scalars.

## TheRunCreatesTheFolderOfItsCheckpoint

The checkpoint is named two folders deep under a temporary folder that does not exist. The run
trains, and the checkpoint file exists afterwards; the test removes the folder.

## TheAnchorReportReadsTheCheckpointOfTheRun

A run is trained as above, and the report loads its checkpoint. The report names one head and two
classes, ends the detections with "detections over 4 images.", and prints the mean average
precision line over 4 images, which only a split with a `boxes` role reaches. The test states
no detection count and no precision: two updates of an untuned model fix neither.
