# tests/applications/programs/source/test_training_program.cpp

Suites: `Medium_TrainingProgram`, in `sushiai_programs_tests`.

Runs the `anchor` row of the run table over a fixture through `run_training_file` of
`training_program.hpp`. `train.cpp` keeps only its arguments and its exit code, so a change to the
assembly reaches this test.

The fixture is `tests/applications/programs/fixtures/tiny_yolo/`: a model file of two
convolutions with one head of two anchors and two classes over a 16 by 16 input, and a
YOLO-layout dataset of four 16 by 16 PNG images with one or two labelled boxes each, under
`images/train` and `labels/train`. The tests read it in place through `SA_TEST_FIXTURE_DIR` and
write only the checkpoint, under the temporary directory. `tiny_yolo_files.hpp` builds the training
file and the evaluation file the cases of this file and of `test_inference_program.cpp` share.

## TheAnchorRowTrainsTwoStepsAndWritesACheckpoint

One epoch of two batches of two images is two updates. The optimizer has taken 2 steps, the run
reports one compile, the checkpoint file exists, and the output prints "compiles: 1". Neither the
header nor the checkpoint line counts zero learnable scalars.

## TheRunCreatesTheFolderOfItsCheckpoint

The checkpoint is named two folders deep under a temporary folder that does not exist. The run
trains, and the checkpoint file exists afterwards; the test removes the folder.

## ACheckpointFolderUnderAFileIsRefusedBeforeTheRunTrains

The checkpoint is named below a regular file, so its folder cannot be created. The run fails with
`InvalidFile`, the message holds "could not create" and "the folder of the checkpoint", and the run
has printed nothing, so it failed before the header.

## AFileWithoutACheckpointTrainsAndWritesNothing

A file whose `checkpoint` is cleared still trains two steps and prints no "wrote " line.

## AFreshAnchorRunStartsTheObjectnessAndClassBiasesFromTheStride

After the run, the checkpoint is loaded into a new setup of the same files and seed. The model has
one anchor head of stride 2. Each of its two anchors holds an objectness bias within 0.5 of
log(8 / 320^2) and a first class bias within 0.5 of log(0.6 / (2 - 0.99)). Two SGD updates at a rate
of 0.001 move a bias by far less than that tolerance.
