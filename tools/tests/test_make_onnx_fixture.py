# test_make_onnx_fixture.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the ONNX fixture writer: its graph, its determinism and the files it writes.

The cases need numpy and onnx with onnx.reference; they are skipped where those are not installed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference import make_onnx_fixture as fixture  # noqa: E402

try:
    fixture._import_onnx()
    K_HAS_ONNX = True
except fixture.FixtureError:
    K_HAS_ONNX = False


@unittest.skipUnless(K_HAS_ONNX, "numpy and onnx with onnx.reference are not installed")
class GraphTest(unittest.TestCase):
    """Checks the operators and the opset of the fixture graph."""

    K_TEST_SIZE = "small"

    def test_the_graph_holds_the_pattern_of_a_yolo_export(self) -> None:
        """Checks the operator of each node, in order."""
        model = fixture.build_model(fixture.draw_weights(fixture.K_DEFAULT_SEED))
        operators = [node.op_type for node in model.graph.node]
        self.assertEqual(
            operators,
            ["Conv", "BatchNormalization", "Sigmoid", "Mul", "Resize", "Concat", "Conv", "Reshape"],
        )

    def test_the_model_states_opset_seventeen(self) -> None:
        """Checks the default-domain opset import."""
        model = fixture.build_model(fixture.draw_weights(fixture.K_DEFAULT_SEED))
        self.assertEqual([(o.domain, o.version) for o in model.opset_import], [("", 17)])


@unittest.skipUnless(K_HAS_ONNX, "numpy and onnx with onnx.reference are not installed")
class OutputTest(unittest.TestCase):
    """Checks that the expected output is the reference evaluator's."""

    K_TEST_SIZE = "small"

    def test_the_output_has_the_shape_of_the_flattened_join(self) -> None:
        """Checks one row of 2 x 8 x 8 values."""
        model = fixture.build_model(fixture.draw_weights(fixture.K_DEFAULT_SEED))
        output = fixture.evaluate(model, fixture.fixed_input())
        self.assertEqual(output.shape, (1, 128))

    def test_the_same_seed_gives_the_same_output(self) -> None:
        """Checks that two builds with one seed agree bit for bit."""
        first = fixture.evaluate(
            fixture.build_model(fixture.draw_weights(3)), fixture.fixed_input()
        )
        second = fixture.evaluate(
            fixture.build_model(fixture.draw_weights(3)), fixture.fixed_input()
        )
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_another_seed_gives_another_output(self) -> None:
        """Checks that the seed reaches the weights."""
        first = fixture.evaluate(
            fixture.build_model(fixture.draw_weights(3)), fixture.fixed_input()
        )
        second = fixture.evaluate(
            fixture.build_model(fixture.draw_weights(4)), fixture.fixed_input()
        )
        self.assertNotEqual(first.tobytes(), second.tobytes())


@unittest.skipUnless(K_HAS_ONNX, "numpy and onnx with onnx.reference are not installed")
class WriteTest(unittest.TestCase):
    """Checks the two files the writer leaves."""

    K_TEST_SIZE = "small"

    def test_the_files_are_written_and_the_model_is_small(self) -> None:
        """Checks both names, and that the model stays under 50 KB."""
        with tempfile.TemporaryDirectory() as folder:
            model_path, expected_path = fixture.write_fixture(Path(folder) / "onnx")
            self.assertEqual(model_path.name, fixture.K_MODEL_FILE)
            self.assertEqual(expected_path.name, fixture.K_EXPECTED_FILE)
            self.assertLess(model_path.stat().st_size, 50 * 1024)

    def test_the_expected_file_states_input_and_output_values(self) -> None:
        """Checks that each tensor lists as many values as its shape holds."""
        with tempfile.TemporaryDirectory() as folder:
            _, expected_path = fixture.write_fixture(Path(folder))
            record = json.loads(expected_path.read_text(encoding="utf-8"))
        for key in ("input", "output"):
            count = 1
            for extent in record[key]["shape"]:
                count *= extent
            self.assertEqual(len(record[key]["values"]), count)

    def test_a_folder_that_cannot_be_made_is_refused(self) -> None:
        """Checks that a file in the place of the folder raises FixtureError."""
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            with self.assertRaises(fixture.FixtureError):
                fixture.write_fixture(blocker / "onnx")


if __name__ == "__main__":
    unittest.main()
