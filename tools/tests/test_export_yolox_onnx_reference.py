# test_export_yolox_onnx_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the runtime-free parts of the YOLOX ONNX detection exporter."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.export_yolox_onnx_reference import (  # noqa: E402
    K_FORMAT_VERSION,
    K_MAX_DETECTIONS,
    ExportError,
    check_output_names,
    corners,
    decode_level,
    detection_row,
    image_name,
    intersection_over_union,
    reference_document,
    sigmoid,
    suppress,
)


def _all_names() -> list[str]:
    """Returns the nine contract names of the head outputs."""
    roles = ("classes", "boxes", "objectness")
    return [f"level{level}.{role}" for level in range(3) for role in roles]


class ImageNameTest(unittest.TestCase):
    """Tests the name of the letterboxed copy of a source image."""

    K_TEST_SIZE = "small"

    def test_a_jpeg_becomes_a_png_below_the_image_folder(self) -> None:
        """A JPEG keeps its stem and is named as a PNG below images/."""
        self.assertEqual(image_name(Path("x/frame_0007.jpg")), "images/frame_0007.png")


class SigmoidTest(unittest.TestCase):
    """Tests the logistic function."""

    K_TEST_SIZE = "small"

    def test_zero_is_one_half_and_the_tails_do_not_overflow(self) -> None:
        """sigmoid(0) is 0.5; a logit of -1000 gives 0 instead of an overflow."""
        self.assertEqual((sigmoid(0.0), sigmoid(-1000.0), sigmoid(1000.0)), (0.5, 0.0, 1.0))

    def test_the_logit_of_three_quarters_gives_three_quarters(self) -> None:
        """sigmoid(ln 3) is 3/4."""
        self.assertAlmostEqual(sigmoid(math.log(3.0)), 0.75, places=12)


class IntersectionOverUnionTest(unittest.TestCase):
    """Tests the overlap measure of two boxes."""

    K_TEST_SIZE = "small"

    def test_half_overlapping_squares_give_one_third(self) -> None:
        """Two 2x2 squares that share a 1x2 strip: 2 / (4 + 4 - 2) = 1/3."""
        self.assertAlmostEqual(
            intersection_over_union([0, 0, 2, 2], [1, 0, 3, 2]), 1.0 / 3.0, places=12,
        )

    def test_disjoint_boxes_give_zero(self) -> None:
        """Boxes that only touch share no area."""
        self.assertEqual(intersection_over_union([0, 0, 1, 1], [1, 0, 2, 1]), 0.0)


class CornersTest(unittest.TestCase):
    """Tests the move from centre and extent to corners."""

    K_TEST_SIZE = "small"

    def test_centre_and_extent_give_the_four_corners(self) -> None:
        """Centre (10, 20), width 4 and height 6 span (8, 17) to (12, 23)."""
        self.assertEqual(corners(10.0, 20.0, 4.0, 6.0), [8.0, 17.0, 12.0, 23.0])


class DecodeLevelTest(unittest.TestCase):
    """Tests the grid decoding of one level."""

    K_TEST_SIZE = "small"

    def test_the_cell_offset_and_the_stride_place_the_box(self) -> None:
        """Cell (x=1, y=0) of a 1x2 grid, stride 8, offsets 0.5 and 0.25, extents exp(0)."""
        boxes = [0.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0]
        found = decode_level(boxes, [-20.0, 20.0], [-20.0, 20.0], (1, 2), 8, 0.5)

        self.assertEqual(len(found), 1)
        score, cls, box = found[0]
        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertEqual(cls, 0)
        self.assertEqual(box, corners(12.0, 2.0, 8.0, 8.0))

    def test_the_best_class_of_an_anchor_is_kept(self) -> None:
        """Class logits (-20, 20) over one anchor keep class 1 with its own probability."""
        found = decode_level([0.0] * 4, [20.0], [-20.0, 20.0], (1, 1), 8, 0.5)

        self.assertEqual([cls for _, cls, _ in found], [1])

    def test_the_score_is_objectness_times_class_probability(self) -> None:
        """Both logits 0 give 0.5 * 0.5 = 0.25, which a threshold of 0.25 does not exceed."""
        passes = decode_level([0.0] * 4, [0.0], [0.0], (1, 1), 8, 0.24)
        stops = decode_level([0.0] * 4, [0.0], [0.0], (1, 1), 8, 0.25)

        self.assertEqual((len(passes), len(stops)), (1, 0))
        self.assertEqual(passes[0][0], 0.25)

    def test_arrays_that_do_not_fit_the_grid_are_refused(self) -> None:
        """Two anchors of boxes for a one-cell grid raise ExportError."""
        with self.assertRaises(ExportError):
            decode_level([0.0] * 8, [0.0], [0.0], (1, 1), 8, 0.1)


class SuppressTest(unittest.TestCase):
    """Tests the per-class suppression."""

    K_TEST_SIZE = "small"

    def test_a_lower_score_of_the_same_class_over_the_threshold_is_dropped(self) -> None:
        """Identical boxes of one class keep only the higher score."""
        kept = suppress([(0.5, 0, [0, 0, 2, 2]), (0.9, 0, [0, 0, 2, 2])], 0.7)

        self.assertEqual([score for score, _, _ in kept], [0.9])

    def test_boxes_of_different_classes_do_not_suppress_each_other(self) -> None:
        """Identical boxes of two classes both stay, highest score first."""
        kept = suppress([(0.5, 1, [0, 0, 2, 2]), (0.9, 0, [0, 0, 2, 2])], 0.7)

        self.assertEqual([cls for _, cls, _ in kept], [0, 1])

    def test_an_overlap_equal_to_the_threshold_does_not_suppress(self) -> None:
        """An IoU of exactly 1/3 against a threshold of 1/3 keeps both boxes."""
        kept = suppress([(0.9, 0, [0, 0, 2, 2]), (0.5, 0, [1, 0, 3, 2])], 1.0 / 3.0)

        self.assertEqual(len(kept), 2)

    def test_no_more_than_the_detection_limit_are_kept(self) -> None:
        """K_MAX_DETECTIONS + 5 disjoint boxes leave K_MAX_DETECTIONS."""
        count = K_MAX_DETECTIONS + 5
        boxes = [(1.0 - i / 1000.0, 0, [3.0 * i, 0, 3.0 * i + 2, 2]) for i in range(count)]

        self.assertEqual(len(suppress(boxes, 0.7)), K_MAX_DETECTIONS)


class DetectionRowTest(unittest.TestCase):
    """Tests the row one detection is written as."""

    K_TEST_SIZE = "small"

    def test_the_class_comes_first_then_the_score_then_the_corners(self) -> None:
        """Class 3, score 0.5 and corners 1 to 4 are six floats in that order."""
        self.assertEqual(detection_row(3, 0.5, [1, 2, 3, 4]), [3.0, 0.5, 1.0, 2.0, 3.0, 4.0])


class OutputNamesTest(unittest.TestCase):
    """Tests the check of the model's output names."""

    K_TEST_SIZE = "small"

    def test_the_nine_contract_names_pass(self) -> None:
        """A model that publishes all nine names is accepted."""
        check_output_names(_all_names())

    def test_a_missing_name_is_reported(self) -> None:
        """Without level2.objectness the error names it."""
        names = [name for name in _all_names() if name != "level2.objectness"]
        with self.assertRaises(ExportError) as raised:
            check_output_names(names)

        self.assertIn("level2.objectness", str(raised.exception))


class ReferenceDocumentTest(unittest.TestCase):
    """Tests the document of one model."""

    K_TEST_SIZE = "small"

    def test_the_document_states_how_the_images_were_prepared(self) -> None:
        """The size, the normalisation, the thresholds and the format version are written."""
        document = reference_document(
            "yolox_x.onnx", "onnx.reference 1.0", (800, 1440), (0.5, 0.5, 0.5),
            (0.25, 0.25, 0.25), 0.1, 0.7, [],
        )

        self.assertEqual(
            (
                document["format_version"],
                document["input_size"],
                document["mean"],
                document["score_threshold"],
                document["suppression_threshold"],
                document["onnx"],
            ),
            (K_FORMAT_VERSION, [800, 1440], [0.5, 0.5, 0.5], 0.1, 0.7, "yolox_x.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
