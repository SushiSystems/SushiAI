# test_export_yolox_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the torch-free parts of the YOLOX reference exporter.

Covers the port names, the letterbox geometry, the NCHW to NHWC layout, the head records and the
files written. The run under PyTorch is not tested here: it needs `yolox` and a checkpoint.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.export_yolox_reference import (  # noqa: E402
    K_CONTEXT_FILE,
    K_INPUT_NAME,
    K_REFERENCE_FILE,
    ExportError,
    head_records,
    input_record,
    letterbox_geometry,
    nchw_to_nhwc,
    port_name,
    reference_context,
    write_reference,
)


def _levels(rows: tuple[int, int, int], columns: tuple[int, int, int]) -> list[dict]:
    """Returns three levels whose values count up from 0 within each output."""
    levels = []
    for level in range(3):
        outputs = {}
        for role, channels in (("classes", 1), ("boxes", 4), ("objectness", 1)):
            shape = (1, channels, rows[level], columns[level])
            count = channels * rows[level] * columns[level]
            outputs[role] = (shape, [float(v) for v in range(count)])
        levels.append(outputs)
    return levels


class PortNameTest(unittest.TestCase):
    """Checks the names the model file and the inference row agree on."""

    K_TEST_SIZE = "small"

    def test_name_is_level_then_role(self) -> None:
        """Checks the three roles of the last level."""
        self.assertEqual(port_name(2, "classes"), "level2.classes")
        self.assertEqual(port_name(2, "boxes"), "level2.boxes")
        self.assertEqual(port_name(2, "objectness"), "level2.objectness")

    def test_name_refuses_a_fourth_level(self) -> None:
        """Checks that a level past the third is refused."""
        with self.assertRaises(ExportError):
            port_name(3, "classes")

    def test_name_refuses_an_unknown_role(self) -> None:
        """Checks that a role the head lacks is refused."""
        with self.assertRaises(ExportError):
            port_name(0, "scores")


class LetterboxGeometryTest(unittest.TestCase):
    """Checks the size YOLOX resizes to before it pads."""

    K_TEST_SIZE = "small"

    def test_a_wide_image_fits_by_its_width_ratio(self) -> None:
        """Checks 800x2880 into 800x1440: ratios 1.0 and 0.5, so 0.5 and 400x1440."""
        geometry = letterbox_geometry(800, 2880, (800, 1440))
        self.assertEqual(geometry.ratio, 0.5)
        self.assertEqual((geometry.height, geometry.width), (400, 1440))

    def test_a_tall_image_fits_by_its_height_ratio(self) -> None:
        """Checks 1600x1440 into 800x1440: ratios 0.5 and 1.0, so 0.5 and 800x720."""
        geometry = letterbox_geometry(1600, 1440, (800, 1440))
        self.assertEqual(geometry.ratio, 0.5)
        self.assertEqual((geometry.height, geometry.width), (800, 720))

    def test_a_non_positive_extent_is_refused(self) -> None:
        """Checks that a zero-height image is refused."""
        with self.assertRaises(ExportError):
            letterbox_geometry(0, 10, (800, 1440))


class LayoutTest(unittest.TestCase):
    """Checks the NCHW to NHWC move against the index formula."""

    K_TEST_SIZE = "small"

    def test_two_channels_of_a_two_by_three_grid(self) -> None:
        """Checks element (h, w, c) is source element (c * 2 + h) * 3 + w."""
        values = [float(v) for v in range(12)]
        moved = nchw_to_nhwc(values, (1, 2, 2, 3))
        expected = [
            values[(c * 2 + h) * 3 + w] for h in range(2) for w in range(3) for c in range(2)
        ]
        self.assertEqual(moved, expected)
        self.assertEqual(moved[:4], [0.0, 6.0, 1.0, 7.0])

    def test_a_short_array_is_refused(self) -> None:
        """Checks that values that do not fill the shape are refused."""
        with self.assertRaises(ExportError):
            nchw_to_nhwc([0.0] * 5, (1, 2, 2, 3))


class HeadRecordsTest(unittest.TestCase):
    """Checks the nine records of a head."""

    K_TEST_SIZE = "small"

    def test_nine_records_in_level_then_role_order(self) -> None:
        """Checks the names and the NHWC shapes of every record."""
        records = head_records(_levels((4, 2, 1), (6, 3, 2)))
        self.assertEqual(
            [record.name for record in records],
            [
                f"level{level}.{role}"
                for level in range(3)
                for role in ("classes", "boxes", "objectness")
            ],
        )
        self.assertEqual(records[0].shape, (1, 4, 6, 1))
        self.assertEqual(records[1].shape, (1, 4, 6, 4))
        self.assertEqual(records[8].shape, (1, 1, 2, 1))

    def test_a_box_record_holds_the_values_channel_last(self) -> None:
        """Checks the first four floats of the finest box record: cell (0, 0), four channels."""
        records = head_records(_levels((4, 2, 1), (6, 3, 2)))
        floats = struct.unpack_from("<4f", records[1].payload, 0)
        self.assertEqual(floats, (0.0, 24.0, 48.0, 72.0))

    def test_a_level_with_two_classes_is_refused(self) -> None:
        """Checks that a classes output of two channels is refused for the one-class model."""
        levels = _levels((4, 2, 1), (6, 3, 2))
        levels[0]["classes"] = ((1, 2, 4, 6), [0.0] * 48)
        with self.assertRaises(ExportError):
            head_records(levels)

    def test_two_levels_are_refused(self) -> None:
        """Checks that a head with two levels is refused."""
        with self.assertRaises(ExportError):
            head_records(_levels((4, 2, 1), (6, 3, 2))[:2])

    def test_outputs_on_different_grids_are_refused(self) -> None:
        """Checks that a level whose boxes sit on another grid than its classes is refused."""
        levels = _levels((4, 2, 1), (6, 3, 2))
        levels[1]["boxes"] = ((1, 4, 2, 4), [0.0] * 32)
        with self.assertRaises(ExportError):
            head_records(levels)

    def test_a_missing_role_is_refused(self) -> None:
        """Checks that a level without objectness is refused."""
        levels = _levels((4, 2, 1), (6, 3, 2))
        del levels[2]["objectness"]
        with self.assertRaises(ExportError):
            head_records(levels)


class WriteReferenceTest(unittest.TestCase):
    """Checks the files the exporter writes."""

    K_TEST_SIZE = "small"

    def test_the_sacp_file_and_its_context_are_written(self) -> None:
        """Checks the SACP magic, the record count, and the context's outputs."""
        records = [input_record((1, 3, 2, 2), [float(v) for v in range(12)])]
        records.extend(head_records(_levels((4, 2, 1), (6, 3, 2))))
        geometry = letterbox_geometry(800, 2880, (800, 1440))
        context = reference_context(
            Path("mot17x.pth.tar"),
            Path("frame.jpg"),
            (800, 1440),
            geometry,
            None,
            None,
            "0.1.0",
        )
        with tempfile.TemporaryDirectory() as folder:
            written = write_reference(Path(folder) / "out", records, context)
            data = (Path(folder) / "out" / K_REFERENCE_FILE).read_bytes()
            described = json.loads(
                (Path(folder) / "out" / K_CONTEXT_FILE).read_text(encoding="utf-8"),
            )

        self.assertEqual([path.name for path in written], [K_REFERENCE_FILE, K_CONTEXT_FILE])
        self.assertEqual(struct.unpack_from("<IIII", data, 0), (0x50434153, 2, 10, 0))
        self.assertEqual(len(described["outputs"]), 9)
        self.assertEqual(described["outputs"][0], "level0.classes")
        self.assertEqual(described["resized"], [400, 1440])
        self.assertIsNone(described["mean"])

    def test_the_input_record_is_laid_out_nhwc(self) -> None:
        """Checks the input record's shape and its first pixel."""
        record = input_record((1, 3, 2, 2), [float(v) for v in range(12)])
        self.assertEqual(record.name, K_INPUT_NAME)
        self.assertEqual(record.shape, (1, 2, 2, 3))
        self.assertEqual(struct.unpack_from("<3f", record.payload, 0), (0.0, 4.0, 8.0))


if __name__ == "__main__":
    unittest.main()
