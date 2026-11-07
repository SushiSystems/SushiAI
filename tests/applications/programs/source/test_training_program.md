# tests/applications/programs/source/test_training_program.cpp

Suites: `Medium_TrainingProgram`, in `sushiai_programs_tests`.

Runs the `anchor` row of the run table over a fixture through `run_training_file` of
`training_program.hpp`. `train.cpp` keeps only its arguments and its exit code, so a change to the
assembly reaches this test.

The fixture is `tests/applications/programs/fixtures/tiny_yolo/`: a model file of two
convolutions with one head of two anchors and two classes over a 16 by 16 input, and a
YOLO-layout dataset of four 16 by 16 PNG images with one or two labelled boxes each, under
`images/train` and `labels/train`. The tests read it in place through `SA_TEST_FIXTURE_DIR` and
write only the checkpoint, under a scratch folder or file. `tiny_yolo_files.hpp` builds the training
file and the evaluation file the cases of this file and of `test_inference_program.cpp` share.

## TheAnchorRowTakesTwoOptimizerSteps

One epoch of two batches of two images is two updates. The optimizer has taken 2 steps.

## TheAnchorRowCompilesOnce

The run reports one compile, and the output prints "compiles: 1".

## TheAnchorRowWritesItsCheckpoint

The checkpoint file the training file names exists after the run.

## TheAnchorRowPrintsNoZeroScalarCount

Neither the header nor the checkpoint line counts zero learnable scalars.

## TheRunCreatesTheFolderOfItsCheckpoint

The checkpoint is named two folders deep under a scratch folder that does not hold them yet. The
run trains, and the checkpoint file exists afterwards.

## ACheckpointFolderUnderAFileIsRefusedBeforeTheRunTrains

The checkpoint is named below a regular file, so its folder cannot be created. The run fails with
`InvalidFile`, the message holds "could not create" and "the folder of the checkpoint", and the run
has printed nothing, so it failed before the header.

## AFileWithoutACheckpointTrainsItsSteps

A file whose `checkpoint` is cleared still trains two steps.

## AFileWithoutACheckpointPrintsNoWroteLine

A file whose `checkpoint` is cleared prints no "wrote " line.
