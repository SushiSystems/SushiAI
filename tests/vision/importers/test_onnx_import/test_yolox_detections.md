# tests/vision/importers/test_onnx_import/test_yolox_detections.cpp

Suites: `Large_YoloxOnnxDetections`.

The folder `SA_YOLOX_DATA_DIR` names holds `yolox_x.onnx` and the `references/` folder that
`tools/reference/export_yolox_onnx_reference.py` writes: each image fitted into the network input
(top left, padded with 114) as a PNG, and `yolox_x_onnx.json` with the normalisation, the two
thresholds and the detections the reference runtime made on the images. The runtime is onnxruntime
or, when that is not installed, `onnx.reference.ReferenceEvaluator`; the file records which. The test
skips when the variable is unset or a file is missing.

The ONNX file must publish the nine raw head outputs as `level<l>.classes`, `level<l>.boxes` and
`level<l>.objectness`, the names `yolox_head_contract.hpp` fixes. The test imports it with
`import_onnx_files`, runs it, and decodes and suppresses the outputs with
`select_yolox_detections` at the thresholds the reference file states, as `sa inference` runs a
YOLOX model. It applies `(byte / 255 - mean) / std` with the file's mean and standard deviation.

## TheDetectionsAreTheReferenceRuntimesOwn

On every reference image the imported file finds as many detections as the reference runtime, of the
same classes; sorted by class and score, each score is within 2048 u and each corner within 2048 u
of the larger input side, the row bound of the Ultralytics detection test. Nothing is clipped: the
reference does not clip either.

The case checks what the Small and Medium ONNX cases cannot: a whole YOLOX-X graph through the
importer, the operator rows of its real export, and the head's decode, against another runtime.
