# check_exported_files.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Reads the files `sushiai export` wrote with the readers of other frameworks.

It needs three readers: onnx (onnx.checker and onnx.reference.ReferenceEvaluator) for .onnx,
torch (torch.load with weights_only=True) for .pth, and safetensors for .safetensors. OpenCV
(cv2.dnn), when installed, also reads a .cfg with its .weights and a .prototxt with its
.caffemodel. Usage: see tools/reference/README.md, "check_exported_files.py".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

K_ONNX_SUFFIX = ".onnx"
K_TORCH_SUFFIXES = (".pth", ".pt")
K_SAFETENSORS_SUFFIX = ".safetensors"
K_DARKNET_SUFFIX = ".cfg"
K_DARKNET_WEIGHTS_SUFFIX = ".weights"
K_CAFFE_SUFFIX = ".prototxt"
K_CAFFE_WEIGHTS_SUFFIX = ".caffemodel"
K_DEFAULT_TOLERANCE = 1e-5


class CheckError(Exception):
    """Reports a file a reader refused, or a reader that is not installed."""


@dataclass(frozen=True)
class CheckedFile:
    """Holds what one reader said of one file; `reader` is empty for a skipped file."""

    path: Path
    reader: str
    summary: str


def _load_expected(path: Path) -> dict[str, Any]:
    """Returns the expected document: `input` and `output`, each with `name`, `shape`, `values`."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckError(f"{path}: cannot be read as JSON: {error}") from error
    for key in ("input", "output"):
        entry = document.get(key) if isinstance(document, dict) else None
        if not isinstance(entry, dict) or not {"name", "shape", "values"} <= entry.keys():
            raise CheckError(f"{path}: '{key}' is not an object with name, shape and values")
    return document


def check_onnx(path: Path, expected: dict[str, Any] | None, tolerance: float) -> CheckedFile:
    """Checks `path` with onnx.checker and, given `expected`, runs ReferenceEvaluator on it."""
    try:
        import numpy
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError as error:
        raise CheckError(f"{path}: the onnx reader is not installed: {error}") from error
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:  # onnx raises several types for one refusal
        raise CheckError(f"{path}: onnx refused the file: {error}") from error

    summary = f"onnx.checker passed, {len(model.graph.node)} nodes"
    if expected is None:
        return CheckedFile(path, "onnx", summary)

    stated = expected["input"]
    image = numpy.asarray(stated["values"], dtype=numpy.float32).reshape(stated["shape"])
    try:
        outputs = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: image})
    except Exception as error:  # the evaluator raises the error of the operator that failed
        raise CheckError(f"{path}: ReferenceEvaluator failed: {error}") from error
    wanted = numpy.asarray(expected["output"]["values"], dtype=numpy.float32)
    computed = numpy.asarray(outputs[0], dtype=numpy.float32).reshape(-1)
    if computed.size != wanted.size:
        raise CheckError(f"{path}: the output holds {computed.size} values, "
                         f"the expected file {wanted.size}")
    largest = float(numpy.max(numpy.abs(computed - wanted))) if wanted.size else 0.0
    if largest > tolerance:
        raise CheckError(f"{path}: the output differs from the expected file by up to {largest}")
    return CheckedFile(path, "onnx", f"{summary}; the output equals the expected file within "
                                     f"{largest:.3g}")


def check_torch(path: Path) -> CheckedFile:
    """Loads `path` with torch.load(weights_only=True) and counts its tensors."""
    try:
        import torch
    except ImportError as error:
        raise CheckError(f"{path}: the torch reader is not installed: {error}") from error
    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:  # torch raises several types for one refusal
        raise CheckError(f"{path}: torch.load refused the file: {error}") from error

    tensors = _count_tensors(loaded, torch.Tensor)
    return CheckedFile(path, "torch", f"torch.load passed, {tensors} tensors")


def _count_tensors(value: Any, tensor_type: type) -> int:
    """Returns how many tensors `value` holds, looking into dictionaries."""
    if isinstance(value, tensor_type):
        return 1
    if isinstance(value, dict):
        return sum(_count_tensors(item, tensor_type) for item in value.values())
    return 0


def check_safetensors(path: Path) -> CheckedFile:
    """Opens `path` with safetensors and reads every tensor."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise CheckError(f"{path}: the safetensors reader is not installed: {error}") from error
    try:
        tensors = load_file(str(path))
    except Exception as error:  # safetensors raises its own error types
        raise CheckError(f"{path}: safetensors refused the file: {error}") from error
    return CheckedFile(path, "safetensors", f"safetensors passed, {len(tensors)} tensors")


def _check_with_opencv(path: Path, weights_suffix: str, reader: str,
                       read_net: Callable[[Any, str, str], Any]) -> CheckedFile:
    """Reads `path` and the weight file beside it with an OpenCV reader, or says it is skipped."""
    try:
        import cv2
    except ImportError:
        return CheckedFile(path, "", "OpenCV is not installed")
    weights = path.with_suffix(weights_suffix)
    weights_text = str(weights) if weights.is_file() else ""
    try:
        net = read_net(cv2, str(path), weights_text)
    except Exception as error:  # cv2.error is the only type, but it is not importable alone
        raise CheckError(f"{path}: {reader} refused the file: {error}") from error
    layers = len(net.getLayerNames())
    with_weights = f" with {weights.name}" if weights_text else " without weights"
    return CheckedFile(path, reader, f"{reader} passed{with_weights}, {layers} layers")


def check_darknet(path: Path) -> CheckedFile:
    """Reads a .cfg and its .weights with cv2.dnn.readNetFromDarknet."""
    return _check_with_opencv(path, K_DARKNET_WEIGHTS_SUFFIX, "cv2.dnn.readNetFromDarknet",
                              lambda cv2, cfg, weights: cv2.dnn.readNetFromDarknet(cfg, weights))


def check_caffe(path: Path) -> CheckedFile:
    """Reads a .prototxt and its .caffemodel with cv2.dnn.readNetFromCaffe."""
    return _check_with_opencv(path, K_CAFFE_WEIGHTS_SUFFIX, "cv2.dnn.readNetFromCaffe",
                              lambda cv2, net, weights: cv2.dnn.readNetFromCaffe(net, weights))


def collect_files(paths: list[Path]) -> list[Path]:
    """Returns the files named, and the files of each folder named, in name order."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.iterdir() if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise CheckError(f"{path}: no such file or folder")
    return files


def check_file(path: Path, expected: dict[str, Any] | None,
               tolerance: float) -> CheckedFile | None:
    """Checks `path` with the reader of its suffix; returns None for a file no reader takes."""
    suffix = path.suffix.lower()
    if suffix == K_ONNX_SUFFIX:
        return check_onnx(path, expected, tolerance)
    if suffix in K_TORCH_SUFFIXES:
        return check_torch(path)
    if suffix == K_SAFETENSORS_SUFFIX:
        return check_safetensors(path)
    if suffix == K_DARKNET_SUFFIX:
        return check_darknet(path)
    if suffix == K_CAFFE_SUFFIX:
        return check_caffe(path)
    return None


def main(argv: list[str]) -> int:
    """Checks every file the arguments name and returns 1 when a reader refused one."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n")[1])
    parser.add_argument("paths", type=Path, nargs="+",
                        help="exported files, or folders whose files are all checked")
    parser.add_argument("--expected", type=Path,
                        help="a JSON file with the input of the .onnx and the output it must "
                             "give, as tools/reference/make_onnx_fixture.py writes one")
    parser.add_argument("--tolerance", type=float, default=K_DEFAULT_TOLERANCE,
                        help="the largest difference an output value may have; default 1e-5")
    arguments = parser.parse_args(argv)

    refused: list[str] = []
    try:
        expected = _load_expected(arguments.expected) if arguments.expected else None
        files = collect_files(arguments.paths)
    except CheckError as error:
        print(f"check_exported_files: {error}", file=sys.stderr)
        return 1

    for path in files:
        try:
            checked = check_file(path, expected, arguments.tolerance)
        except CheckError as error:
            refused.append(str(path))
            print(f"refused {error}")
            continue
        if checked is None:
            print(f"skipped {path}: no reader takes its suffix alone")
        elif not checked.reader:
            print(f"skipped {checked.path}: {checked.summary}")
        else:
            print(f"read    {checked.path}: {checked.summary}")

    if refused:
        print(f"check_exported_files: refused {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
