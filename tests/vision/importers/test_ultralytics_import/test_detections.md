# tests/vision/importers/test_ultralytics_import/test_detections.cpp

Suites: `Large_UltralyticsDetections`, one row per released checkpoint: `yolo26n`, `yolov8n`,
`yolo11n` and `yolov5nu`.

The folder `SA_ULTRALYTICS_DATA_DIR` names holds `<stem>.pt` and the `references/` folder that
`tools/reference/export_ultralytics_detections.py` writes: sixteen coco128 images resized to
640x640 as PNG, and one `<stem>.json` per checkpoint with Ultralytics' detections on them at
score 0.25 and suppression 0.7. A row is skipped when the variable is unset or a file is missing.

The test imports the checkpoint with `import_ultralytics_files`, runs it as `sa inference` runs an
anchor-free model (one-to-many outputs, `FreeBatchDecoder`, suppression by best class at the
task's threshold) and clips each corner to the image, as Ultralytics does (KI-22).

## TheDetectionsAreUltralyticsOwn

On every reference image the checkpoint finds as many detections as Ultralytics, of the same
classes; sorted by class and score, each score is within 2048 u and each corner within 2048 u of
the side of 640, the row bound of the Darknet parity cases.
