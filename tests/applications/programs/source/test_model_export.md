# tests/applications/programs/source/test_model_export.cpp

Suites: `Small_ModelExport`.

Tests `export_model_files` of `applications/programs/source/model_export.hpp` and
`export_weights_file` of `weights_export.hpp` through `compose_registries`, as `sushiai export`
runs them. Each case imports a file of another framework, writes SushiAI's three files into a
scratch folder, exports them and imports the exported files back. The ONNX and Caffe fixtures are
the importers' own, under `tests/vision/importers/fixtures/`; the Darknet case writes a small
`.cfg` without weights.

## TheCaffeFixtureComesBackThroughTheRegistry

`lenet.prototxt` with `lenet.caffemodel` comes back the same way.

## ADarknetConfigComesBackThroughTheRegistry

The small `.cfg` comes back with the same model text.

## AnUnknownFormatNamesTheExportersAndWritesNoFile

The format `tflite` fails with `InvalidDescription` naming `onnx`, `caffe` and `darknet`, and no
file is written.

## ASecondExportIsRefusedUnlessForced

An export with `--name small` writes `small.cfg`; a second one fails with `Incompatible` naming
it, and one with `ExistingFiles::Replace` passes.

## TheFixtureWeightsComeBackFromEachWeightFormat

The weights of `silu_resize.onnx` go to safetensors and to the PyTorch state dictionary, come back
through `import_weights_file` with an empty map, and equal the imported records bit for bit.
