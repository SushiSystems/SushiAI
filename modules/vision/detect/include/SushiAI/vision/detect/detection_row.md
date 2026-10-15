# modules/vision/detect/include/SushiAI/vision/detect/detection_row.hpp


A detection is six floats: x1, y1, x2, y2, score, class index (`U/nn/modules/head.py:259`).
`K_MAX_DETECTIONS` is `Detect.max_det = 300` (`U/nn/modules/head.py:83`).
`K_VALIDATION_CONFIDENCE` is 0.001, the confidence the validator sets when none is given
(`U/engine/validator.py:144-145`); both validation paths use it.
