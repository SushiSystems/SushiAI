# tests/applications/programs/source/test_inference_program.cpp

Suites: `Medium_InferenceProgram`, in `sushiai_programs_tests`.

Runs the `anchor_detection` row of the report table through `run_inference_file` of
`inference_program.hpp`. `inference.cpp` keeps only its arguments and its exit code, so a change to
the assembly reaches this test. The fixture is the `tiny_yolo` fixture described in
`test_training_program.md`, trained by `run_training_file` for the case that needs a checkpoint.

## TheAnchorReportReadsTheCheckpointOfTheRun

A run is trained, and the report loads its checkpoint. The report names one head and two classes,
ends the detections with "detections over 4 images.", prints the mean average precision line over
4 images, which only a split with a `boxes` role reaches, and ends with "compiles: 1". The test
states no detection count and no precision: two updates of an untuned model fix neither.

## ACheckpointThatDoesNotExistFailsTheReport

A checkpoint path that names no file makes `run_inference_file` return a failure.
