# test_export_caffe_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the Caffe reference exporter: its input, its record and its refusal of OpenCV 5.

The run under OpenCV needs an opencv-python 4.x build; that case is skipped where cv2 has no
readNetFromCaffe.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference import export_caffe_reference as exporter  # noqa: E402
from reference import make_caffe_lenet_fixture as fixture  # noqa: E402

try:
    import numpy
    K_HAS_NUMPY = True
except ImportError:
    K_HAS_NUMPY = False

try:
    import cv2
    K_HAS_READER = hasattr(cv2.dnn, "readNetFromCaffe")
except ImportError:
    K_HAS_READER = False


@unittest.skipUnless(K_HAS_NUMPY, "numpy is not installed")
class InputTest(unittest.TestCase):
    """Checks the fixed input and the record built over it."""

    K_TEST_SIZE = "small"

    def test_the_input_counts_up_from_zero_below_one(self) -> None:
        """Checks the shape, the first value and that no value reaches 1."""
        image = exporter.fixed_input()
        self.assertEqual(image.shape, fixture.K_INPUT_SHAPE)
        self.assertEqual(float(image.reshape(-1)[0]), 0.0)
        self.assertLess(float(image.max()), 1.0)

    def test_the_record_lists_as_many_values_as_each_shape_holds(self) -> None:
        """Checks the input and both outputs of a record over zeros."""
        outputs = {
            name: numpy.zeros((1, fixture.K_CLASSES), dtype=numpy.float32)
            for name in exporter.K_OUTPUT_NAMES
        }
        record = exporter.reference_record(outputs)
        self.assertEqual(len(record["input"]["values"]), 28 * 28)
        self.assertEqual([len(o["values"]) for o in record["outputs"]], [10, 10])


class RefusalTest(unittest.TestCase):
    """Checks what the exporter does without a Caffe reader."""

    K_TEST_SIZE = "small"

    def test_an_opencv_without_the_caffe_reader_is_refused(self) -> None:
        """Checks that a cv2 whose dnn lacks readNetFromCaffe raises ExportError."""
        stub = types.SimpleNamespace(dnn=types.SimpleNamespace(), __version__="5.0.0")
        with mock.patch.dict(sys.modules, {"cv2": stub}):
            with self.assertRaises(exporter.ExportError):
                exporter._import_opencv()

    def test_a_folder_without_the_fixture_is_refused(self) -> None:
        """Checks that the missing prototxt raises ExportError naming the writer."""
        reader = types.SimpleNamespace(readNetFromCaffe=lambda *_: None)
        stub = types.SimpleNamespace(dnn=reader, __version__="4.12.0")
        with mock.patch.dict(sys.modules, {"cv2": stub}), tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(exporter.ExportError) as caught:
                exporter.run_reference(Path(folder))
        self.assertIn("make_caffe_lenet_fixture.py", str(caught.exception))


@unittest.skipUnless(K_HAS_READER, "opencv-python 4.x is not installed")
class RunTest(unittest.TestCase):
    """Checks the run under OpenCV on a fixture written for the test."""

    K_TEST_SIZE = "medium"

    def test_the_probabilities_sum_to_one(self) -> None:
        """Checks the Softmax identity on the output OpenCV computes."""
        with tempfile.TemporaryDirectory() as folder:
            fixture.write_fixture(Path(folder))
            probabilities = exporter.run_reference(Path(folder))["prob"]
        self.assertEqual(probabilities.shape, (1, fixture.K_CLASSES))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
