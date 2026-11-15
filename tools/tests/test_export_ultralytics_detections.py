# test_export_ultralytics_detections.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the Ultralytics-free parts of the detection exporter: names, rows and the document."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.export_ultralytics_detections import (  # noqa: E402
    K_FORMAT_VERSION,
    K_IMAGE_SIDE,
    detection_row,
    image_name,
    reference_document,
)


class ImageNameTest(unittest.TestCase):
    """Tests the name of the square copy of a source image."""

    K_TEST_SIZE = "small"

    def test_a_jpeg_becomes_a_png_below_the_image_folder(self) -> None:
        """A coco128 JPEG keeps its stem and is named as a PNG below images/."""
        self.assertEqual(image_name(Path("x/000000000009.jpg")), "images/000000000009.png")


class DetectionRowTest(unittest.TestCase):
    """Tests the row one detection is written as."""

    K_TEST_SIZE = "small"

    def test_the_class_comes_first_then_the_score_then_the_corners(self) -> None:
        """Class 3, score 0.5 and corners 1 to 4 are written as six floats in that order."""
        self.assertEqual(detection_row(3, 0.5, [1, 2, 3, 4]), [3.0, 0.5, 1.0, 2.0, 3.0, 4.0])


class ReferenceDocumentTest(unittest.TestCase):
    """Tests the document of one checkpoint."""

    K_TEST_SIZE = "small"

    def test_the_document_names_its_checkpoint_side_and_format(self) -> None:
        """The stem yolov8n names yolov8n.pt; the side and the format version are stated."""
        document = reference_document("yolov8n", "8.4.0", [])

        self.assertEqual(
            (document["checkpoint"], document["side"], document["format_version"]),
            ("yolov8n.pt", K_IMAGE_SIDE, K_FORMAT_VERSION),
        )


if __name__ == "__main__":
    unittest.main()
