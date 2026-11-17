# test_make_caffe_lenet_fixture.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the Caffe fixture writer: its prototxt, the bytes of its caffemodel and its files.

The cases need numpy and the protobuf package; they are skipped where those are not installed.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference import make_caffe_lenet_fixture as fixture  # noqa: E402

try:
    fixture._import_numpy()
    fixture._import_protobuf()
    K_HAS_PACKAGES = True
except fixture.FixtureError:
    K_HAS_PACKAGES = False

K_LAYER_TYPES = [
    "Convolution", "Pooling", "Convolution", "Pooling", "InnerProduct", "ReLU", "InnerProduct",
    "Softmax",
]

# Tags of caffe.proto's NetParameter.name (1) and .layer (100): (1 << 3) | 2 = 0x0A, 802 = 0xA2 0x06.
K_NAME_AND_FIRST_LAYER_TAGS = b"\x0a\x05LeNet\xa2\x06"


class PrototxtTest(unittest.TestCase):
    """Checks the layers the prototxt lists."""

    K_TEST_SIZE = "small"

    def test_the_layers_follow_the_lenet_of_the_caffe_mnist_example(self) -> None:
        """Checks the type of each layer, in order."""
        types = re.findall(r'type: "(\w+)"', fixture.build_prototxt())
        self.assertEqual(types, K_LAYER_TYPES)

    def test_the_first_inner_product_reads_the_pooled_feature_map(self) -> None:
        """Checks 8 channels of 4 by 4 after two convolutions and two pools over 28 by 28."""
        self.assertEqual(fixture.inner_product_inputs(), 8 * 4 * 4)


@unittest.skipUnless(K_HAS_PACKAGES, "numpy and protobuf are not installed")
class CaffemodelTest(unittest.TestCase):
    """Checks the bytes of the caffemodel."""

    K_TEST_SIZE = "small"

    def test_the_net_opens_with_the_tags_caffe_proto_numbers(self) -> None:
        """Checks the name field and the first layer field by their tag bytes."""
        model = fixture.build_caffemodel(fixture.draw_weights(fixture.K_DEFAULT_SEED))
        self.assertTrue(model.startswith(K_NAME_AND_FIRST_LAYER_TAGS))

    def test_the_layers_with_weights_come_back_with_their_blob_shapes(self) -> None:
        """Checks the name and the dimensions of every blob after a parse."""
        model = fixture.build_caffemodel(fixture.draw_weights(fixture.K_DEFAULT_SEED))
        net = fixture.message_classes()["NetParameter"]()
        net.ParseFromString(model)
        shapes = {layer.name: [list(blob.shape.dim) for blob in layer.blobs] for layer in net.layer}
        self.assertEqual(
            shapes,
            {
                "conv1": [[4, 1, 5, 5], [4]],
                "conv2": [[8, 4, 5, 5], [8]],
                "ip1": [[32, 128], [32]],
                "ip2": [[10, 32], [10]],
            },
        )

    def test_the_same_seed_gives_the_same_bytes(self) -> None:
        """Checks that two builds with one seed agree byte for byte."""
        first = fixture.build_caffemodel(fixture.draw_weights(3))
        second = fixture.build_caffemodel(fixture.draw_weights(3))
        self.assertEqual(first, second)

    def test_another_seed_gives_other_bytes(self) -> None:
        """Checks that the seed reaches the weights."""
        first = fixture.build_caffemodel(fixture.draw_weights(3))
        second = fixture.build_caffemodel(fixture.draw_weights(4))
        self.assertNotEqual(first, second)


@unittest.skipUnless(K_HAS_PACKAGES, "numpy and protobuf are not installed")
class WriteTest(unittest.TestCase):
    """Checks the two files the writer leaves."""

    K_TEST_SIZE = "small"

    def test_the_files_are_written_and_stay_small(self) -> None:
        """Checks both names, and that the caffemodel stays under 32 KB."""
        with tempfile.TemporaryDirectory() as folder:
            prototxt_path, model_path = fixture.write_fixture(Path(folder) / "caffe")
            self.assertEqual(prototxt_path.name, fixture.K_PROTOTXT_FILE)
            self.assertEqual(model_path.name, fixture.K_MODEL_FILE)
            self.assertLess(model_path.stat().st_size, 32 * 1024)

    def test_a_folder_that_cannot_be_made_is_refused(self) -> None:
        """Checks that a file in the place of the folder raises FixtureError."""
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            with self.assertRaises(fixture.FixtureError):
                fixture.write_fixture(blocker / "caffe")


if __name__ == "__main__":
    unittest.main()
