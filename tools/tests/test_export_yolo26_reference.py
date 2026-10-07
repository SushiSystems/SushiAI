# test_export_yolo26_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the SACP encoder, the layout rule and the manifest of the YOLO26 reference exporter."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.export_yolo26_reference import (  # noqa: E402
    K_FIXTURE_CEILING_BYTES,
    K_LAYOUT_CONV,
    K_LAYOUT_DEPTHWISE,
    K_LAYOUT_UNSPECIFIED,
    K_LAYOUT_VECTOR,
    K_SACP_MAX_NAME_LENGTH,
    K_SACP_MAX_RANK,
    ExportedTensor,
    FixtureTooLargeError,
    ReferenceFile,
    SacpFormatError,
    SacpRecord,
    build_manifest,
    classify_layout,
    encode_files,
    encode_sacp,
    write_reference_files,
)


def _float_record(name: str, shape: tuple[int, ...], values: list[float]) -> SacpRecord:
    """Returns a float32 record holding the values in order."""
    return SacpRecord(name, "float32", shape, struct.pack(f"<{len(values)}f", *values))


def _reference_file(name: str, record: SacpRecord) -> ReferenceFile:
    """Returns a one-tensor reference file around the record."""
    tensor = ExportedTensor(record, "source.py:1 (f)", "as_is", record.shape, 1.0, 6.0, 3.5)
    return ReferenceFile(name, (tensor,), {"mode": "eval"})


class EncodeSacpTest(unittest.TestCase):
    """Checks the bytes of an encoded file against the layout the C++ reader parses."""

    def test_encode_writes_header_then_records_in_order(self) -> None:
        """Checks every field of a two-record file at its byte offset."""
        first = _float_record("a.weight", (2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        second = SacpRecord("idx", "int64", (2,), struct.pack("<2q", 7, -1))
        data = encode_sacp([first, second])

        self.assertEqual(data[:4], b"SACP")
        self.assertEqual(struct.unpack_from("<III", data, 0), (0x50434153, 1, 2))
        self.assertEqual(struct.unpack_from("<I", data, 12), (8,))
        self.assertEqual(data[16:24], b"a.weight")
        self.assertEqual(struct.unpack_from("<II", data, 24), (1, 2))
        self.assertEqual(struct.unpack_from("<2q", data, 32), (2, 3))
        self.assertEqual(struct.unpack_from("<6f", data, 48), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(struct.unpack_from("<I", data, 72), (3,))
        self.assertEqual(data[76:79], b"idx")
        self.assertEqual(struct.unpack_from("<II", data, 79), (6, 1))
        self.assertEqual(struct.unpack_from("<q", data, 87), (2,))
        self.assertEqual(struct.unpack_from("<2q", data, 95), (7, -1))
        self.assertEqual(len(data), 111)

    def test_encode_writes_int32_with_its_code(self) -> None:
        """Checks the dtype code of an int32 record."""
        data = encode_sacp([SacpRecord("m", "int32", (1,), struct.pack("<i", 1))])
        self.assertEqual(struct.unpack_from("<II", data, 17), (5, 1))

    def test_encode_rejects_payload_of_wrong_size(self) -> None:
        """Checks that a payload shorter than the shape needs is refused."""
        with self.assertRaises(SacpFormatError):
            encode_sacp([SacpRecord("a", "float32", (2, 2), b"\x00" * 12)])

    def test_encode_rejects_unknown_dtype(self) -> None:
        """Checks that a dtype outside the format is refused."""
        with self.assertRaises(SacpFormatError):
            encode_sacp([SacpRecord("a", "bool", (1,), b"\x00")])

    def test_encode_rejects_non_positive_extent(self) -> None:
        """Checks that a zero extent is refused."""
        with self.assertRaises(SacpFormatError):
            encode_sacp([SacpRecord("a", "float32", (0,), b"")])

    def test_encode_rejects_rank_above_limit(self) -> None:
        """Checks that a rank past the reader's limit is refused."""
        shape = (1,) * (K_SACP_MAX_RANK + 1)
        with self.assertRaises(SacpFormatError):
            encode_sacp([SacpRecord("a", "float32", shape, b"\x00" * 4)])

    def test_encode_rejects_name_above_limit(self) -> None:
        """Checks that a name past the reader's limit is refused."""
        name = "n" * (K_SACP_MAX_NAME_LENGTH + 1)
        with self.assertRaises(SacpFormatError):
            encode_sacp([_float_record(name, (1,), [0.0])])

    def test_encode_rejects_repeated_name(self) -> None:
        """Checks that two records under one name are refused."""
        record = _float_record("a", (1,), [0.0])
        with self.assertRaises(SacpFormatError):
            encode_sacp([record, record])


class ClassifyLayoutTest(unittest.TestCase):
    """Checks the layout assigned to each kind of state-dict tensor."""

    def test_classify_names_each_kind_the_spec_covers(self) -> None:
        """Checks dense and depthwise weights and the five per-channel vectors."""
        self.assertEqual(classify_layout("Conv2d", "weight", 1, (16, 3, 3, 3)), K_LAYOUT_CONV)
        self.assertEqual(classify_layout("Conv2d", "weight", 64, (64, 1, 3, 3)), K_LAYOUT_DEPTHWISE)
        self.assertEqual(classify_layout("Conv2d", "bias", 1, (80,)), K_LAYOUT_VECTOR)
        for leaf in ("weight", "bias", "running_mean", "running_var"):
            self.assertEqual(classify_layout("BatchNorm2d", leaf, 1, (16,)), K_LAYOUT_VECTOR)

    def test_classify_leaves_other_kinds_unspecified(self) -> None:
        """Checks a grouped weight that is not depthwise and a tensor of another module."""
        self.assertEqual(classify_layout("Conv2d", "weight", 2, (8, 4, 3, 3)), K_LAYOUT_UNSPECIFIED)
        self.assertEqual(classify_layout("Linear", "weight", 1, (10, 4)), K_LAYOUT_UNSPECIFIED)
        self.assertEqual(classify_layout("DFL", "weight", 1, (1, 16, 1, 1)), K_LAYOUT_UNSPECIFIED)


class ManifestTest(unittest.TestCase):
    """Checks what the manifest records and what the writer puts on disk."""

    def test_manifest_records_digest_size_and_tensor(self) -> None:
        """Checks one file entry against the bytes it describes."""
        files = [_reference_file("forward.sacp", _float_record("x", (2,), [1.0, 6.0]))]
        payloads = encode_files(files)
        manifest = json.loads(build_manifest({"seed": 3}, files, payloads, external=False))
        entry = manifest["files"][0]

        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["seed"], 3)
        self.assertEqual(entry["sha256"], hashlib.sha256(payloads["forward.sacp"]).hexdigest())
        self.assertEqual(entry["size_bytes"], len(payloads["forward.sacp"]))
        self.assertEqual(entry["location"], "output_directory")
        self.assertEqual(entry["tensors"][0]["shape"], [2])
        self.assertEqual(entry["tensors"][0]["max"], 6.0)

    def test_write_puts_weights_in_the_external_directory(self) -> None:
        """Checks that only weights.sacp leaves the output directory."""
        files = [
            _reference_file("weights.sacp", _float_record("w", (1,), [1.0])),
            _reference_file("input.sacp", _float_record("image", (1,), [1.0])),
        ]
        payloads = encode_files(files)
        manifest = build_manifest({}, files, payloads, external=True)
        with tempfile.TemporaryDirectory() as folder:
            output, external = Path(folder) / "out", Path(folder) / "external"
            write_reference_files(payloads, manifest, output, external)
            written = sorted(path.name for path in output.iterdir())
            self.assertEqual(written, ["MANIFEST.json", "input.sacp"])
            self.assertEqual((external / "weights.sacp").read_bytes(), payloads["weights.sacp"])
        self.assertEqual(json.loads(manifest)["files"][0]["location"], "external")

    def test_write_refuses_an_output_above_the_ceiling(self) -> None:
        """Checks that nothing is written when the output directory would pass the ceiling."""
        payloads = {"forward.sacp": b"\x00" * (K_FIXTURE_CEILING_BYTES + 1)}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FixtureTooLargeError):
                write_reference_files(payloads, "{}", Path(folder) / "out", None)
            self.assertFalse((Path(folder) / "out").exists())


if __name__ == "__main__":
    unittest.main()
