# tests/vision/registration/test_registration/test_yolox_x_import.cpp

Suites: `Large_YoloxXImport`.

Imports ByteTrack's `mot17x.pth.tar` into the model of `applications/examples/yolox_x/model.json`
through `applications/examples/yolox_x/map.json`. The folder that holds the checkpoint comes from the
environment variable `SA_YOLOX_DATA_DIR`; the test skips when the variable is unset or the file is
missing, as the Ultralytics detection test does for `SA_ULTRALYTICS_DATA_DIR`. The checkpoint is
about 400 MB and the model state is allocated on the engine, so the test belongs to the `large`
tier.

## TheCheckpointFillsEveryRecordAndNothingIsRefused

The model is built and its state created. The torch source opens the file under the map's state key
`model`, and `import_state_weights` matches it against the state's records. A refusal is a single
`Incompatible` message that names every record the file does not supply, every tensor no record takes
and every shape that differs, so the assertion that the call succeeds is the assertion that nothing
is refused. The test then checks that the import returned one tensor per record, in the state's
order, each with the record's name, its shape and the byte count of its elements.

What the test does not check is the values: the numerical comparison against PyTorch is the job of
the reference file `tools/reference/export_yolox_reference.py` writes.
