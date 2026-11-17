# export_caffe_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Runs OpenCV's readNetFromCaffe on the LeNet fixture and writes what it computes.

One fixed input gives the scores before the Softmax (`ip2`) and the probabilities after it
(`prob`). The run needs opencv-python 4.x; OpenCV 5 has no Caffe reader.
Usage: python tools/reference/export_caffe_reference.py FIXTURE_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_caffe_lenet_fixture import (  # noqa: E402
    K_INPUT_NAME,
    K_INPUT_SHAPE,
    K_MODEL_FILE,
    K_PROTOTXT_FILE,
)

K_EXPECTED_FILE = "lenet.expected.json"
K_FORMAT_VERSION = 1
K_OUTPUT_NAMES = ("ip2", "prob")


class ExportError(Exception):
    """Reports a failure to read the fixture, run OpenCV or write the reference."""


def _import_opencv() -> Any:
    """Returns cv2 with its Caffe reader, or raises ExportError."""
    try:
        import cv2
    except ImportError as error:
        raise ExportError(f"the export needs opencv-python 4.x: {error}") from error
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise ExportError(
            f"OpenCV {cv2.__version__} has no readNetFromCaffe; install an opencv-python 4.x build"
        )
    return cv2


def fixed_input() -> Any:
    """Returns the one input the reference is computed on, in NCHW order."""
    import numpy

    count = 1
    for extent in K_INPUT_SHAPE:
        count *= extent
    return (numpy.arange(count, dtype=numpy.float32) / numpy.float32(count)).reshape(K_INPUT_SHAPE)


def run_reference(directory: Path) -> dict[str, Any]:
    """Returns the output of each layer of K_OUTPUT_NAMES for the fixed input, by name."""
    cv2 = _import_opencv()
    prototxt = directory / K_PROTOTXT_FILE
    model = directory / K_MODEL_FILE
    for path in (prototxt, model):
        if not path.is_file():
            raise ExportError(f"{path} is missing; run make_caffe_lenet_fixture.py first")
    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        outputs = {}
        for name in K_OUTPUT_NAMES:
            net.setInput(fixed_input())
            outputs[name] = net.forward(name)
    except cv2.error as error:
        raise ExportError(f"OpenCV refused the fixture: {error}") from error
    return outputs


def reference_record(outputs: dict[str, Any]) -> dict[str, Any]:
    """Returns the JSON document of the input and each output, values in row-major order."""
    image = fixed_input()
    return {
        "format_version": K_FORMAT_VERSION,
        "input": {
            "name": K_INPUT_NAME,
            "shape": list(image.shape),
            "values": [float(value) for value in image.reshape(-1)],
        },
        "outputs": [
            {
                "name": name,
                "shape": list(outputs[name].shape),
                "values": [float(value) for value in outputs[name].reshape(-1)],
            }
            for name in K_OUTPUT_NAMES
        ],
    }


def write_reference(directory: Path) -> Path:
    """Runs the reference on the fixture in `directory`, writes its record and returns the path."""
    record = reference_record(run_reference(directory))
    path = directory / K_EXPECTED_FILE
    try:
        path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    except OSError as error:
        raise ExportError(f"cannot write {path}: {error}") from error
    return path


def main(argv: list[str]) -> int:
    """Parses the command line, writes the reference and returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture_dir", type=Path, help="holds lenet.prototxt and lenet.caffemodel")
    arguments = parser.parse_args(argv)
    try:
        path = write_reference(arguments.fixture_dir)
    except ExportError as error:
        print(f"export_caffe_reference: {error}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
