# export_yolox_onnx_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Writes the detections a YOLOX ONNX file makes on letterboxed images as a reference JSON file.

The file runs on the CPU in float32 and must publish the nine raw head outputs as
`level<l>.classes`, `level<l>.boxes` and `level<l>.objectness`. Each image is fitted as YOLOX's
`preproc` fits it, normalised per channel, decoded, thresholded and suppressed per class.
Usage: see tools/reference/README.md, "export_yolox_onnx_reference.py".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from export_yolox_reference import (
        K_DEFAULT_SIZE,
        K_PAD_VALUE,
        K_ROLES,
        K_STRIDES,
        ExportError,
        letterbox_geometry,
        port_name,
    )
except ImportError:  # imported as reference.export_yolox_onnx_reference, with tools/ on the path
    from reference.export_yolox_reference import (
        K_DEFAULT_SIZE,
        K_PAD_VALUE,
        K_ROLES,
        K_STRIDES,
        ExportError,
        letterbox_geometry,
        port_name,
    )

K_FORMAT_VERSION = 1
K_DEFAULT_ONNX = "yolox_x.onnx"
K_DEFAULT_MEAN = (0.485, 0.456, 0.406)
K_DEFAULT_STD = (0.229, 0.224, 0.225)
K_SCORE_THRESHOLD = 0.1
K_SUPPRESSION_THRESHOLD = 0.7
K_MAX_DETECTIONS = 300
K_REFERENCE_FOLDER = "references"
K_IMAGE_FOLDER = "images"
K_REFERENCE_NAME = "yolox_x_onnx.json"
K_BOX_VALUES = 4


def image_name(source: Path) -> str:
    """Returns the name, below the reference folder, of the letterboxed copy of @p source.

    Args:
        source: A source image.

    Returns:
        ``images/<stem>.png``; PNG keeps the fitted pixels exactly.
    """
    return f"{K_IMAGE_FOLDER}/{source.stem}.png"


def sigmoid(value: float) -> float:
    """Returns 1 / (1 + exp(-value)) without overflowing for large negative values."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def corners(centre_x: float, centre_y: float, width: float, height: float) -> list[float]:
    """Returns x1, y1, x2, y2 of the box with the given centre and extents."""
    return [
        centre_x - width / 2.0,
        centre_y - height / 2.0,
        centre_x + width / 2.0,
        centre_y + height / 2.0,
    ]


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    """Returns the IoU of two boxes given as x1, y1, x2, y2; 0 when either has no area."""
    overlap_width = min(first[2], second[2]) - max(first[0], second[0])
    overlap_height = min(first[3], second[3]) - max(first[1], second[1])
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0
    intersection = overlap_width * overlap_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def decode_level(
    boxes: Sequence[float],
    objectness: Sequence[float],
    classes: Sequence[float],
    grid: tuple[int, int],
    stride: int,
    score_threshold: float,
) -> list[tuple[float, int, list[float]]]:
    """Returns the candidates of one level: score, class and corners of each anchor that passes.

    Each anchor keeps its best class. The score is sigmoid(objectness) times sigmoid(class logit);
    the centre is (grid + offset) * stride and the extent exp(log extent) * stride.

    Args:
        boxes: The level's raw boxes, NHWC `[1, H, W, 4]` flattened: x, y, log w, log h.
        objectness: The objectness logits, `[1, H, W, 1]` flattened.
        classes: The class logits, `[1, H, W, C]` flattened.
        grid: Rows and columns of the level.
        stride: Input pixels per cell.
        score_threshold: A score must exceed it.

    Raises:
        ExportError: The arrays do not fit the grid, or the class count is not positive.
    """
    height, width = grid
    anchors = height * width
    if anchors <= 0 or len(boxes) != anchors * K_BOX_VALUES or len(objectness) != anchors:
        raise ExportError(f"the level arrays do not fit the grid {height}x{width}")
    if len(classes) == 0 or len(classes) % anchors != 0:
        raise ExportError(f"{len(classes)} class values do not fit {anchors} anchors")
    class_count = len(classes) // anchors
    found: list[tuple[float, int, list[float]]] = []
    for anchor in range(anchors):
        objective = sigmoid(objectness[anchor])
        row = classes[anchor * class_count : (anchor + 1) * class_count]
        best = max(range(class_count), key=lambda index: (row[index], -index))
        score = objective * sigmoid(row[best])
        if score <= score_threshold:
            continue
        raw = boxes[anchor * K_BOX_VALUES : (anchor + 1) * K_BOX_VALUES]
        grid_x, grid_y = anchor % width, anchor // width
        found.append((
            score,
            best,
            corners(
                (grid_x + raw[0]) * stride,
                (grid_y + raw[1]) * stride,
                math.exp(raw[2]) * stride,
                math.exp(raw[3]) * stride,
            ),
        ))
    return found


def suppress(
    candidates: Sequence[tuple[float, int, list[float]]],
    iou_threshold: float,
) -> list[tuple[float, int, list[float]]]:
    """Returns the candidates that survive per-class suppression, highest score first.

    A candidate is dropped when a kept one of its class overlaps it by more than the threshold.
    Equal scores keep their order of arrival, and at most K_MAX_DETECTIONS are kept.
    """
    ordered = sorted(candidates, key=lambda candidate: -candidate[0])
    kept: list[tuple[float, int, list[float]]] = []
    for candidate in ordered:
        if len(kept) == K_MAX_DETECTIONS:
            break
        if all(
            other[1] != candidate[1]
            or intersection_over_union(other[2], candidate[2]) <= iou_threshold
            for other in kept
        ):
            kept.append(candidate)
    return kept


def detection_row(cls: int, score: float, box: Sequence[float]) -> list[float]:
    """Returns one detection as the reference file holds it: class, score, x1, y1, x2, y2."""
    return [float(cls), float(score), *(float(v) for v in box)]


def reference_document(
    onnx_name: str,
    runtime: str,
    size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
    score_threshold: float,
    suppression_threshold: float,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    """Returns the reference file: how the images were prepared and what was detected on them.

    Args:
        onnx_name: The name of the ONNX file the detections come from.
        runtime: The runtime that ran it, with its version.
        size: The network input as (height, width).
        mean: The per-channel mean of the normalisation, red first.
        std: The per-channel standard deviation, red first.
        score_threshold: A score must exceed it.
        suppression_threshold: An IoU above it suppresses the lower score of a class.
        images: One entry per image: its file and its detections, highest score first.
    """
    return {
        "format_version": K_FORMAT_VERSION,
        "onnx": onnx_name,
        "runtime": runtime,
        "input_size": [size[0], size[1]],
        "pad_value": K_PAD_VALUE,
        "mean": list(mean),
        "std": list(std),
        "score_threshold": score_threshold,
        "suppression_threshold": suppression_threshold,
        "images": images,
    }


def check_output_names(names: Sequence[str]) -> None:
    """Checks that the model publishes the nine head outputs by their contract names.

    Raises:
        ExportError: One of the nine names is missing.
    """
    missing = [
        port_name(level, role)
        for level in range(len(K_STRIDES))
        for role in K_ROLES
        if port_name(level, role) not in names
    ]
    if missing:
        raise ExportError(f"the ONNX file publishes no output named {', '.join(missing)}")


def write_letterbox(source: Path, folder: Path, size: tuple[int, int]) -> Path:
    """Writes @p source fitted into @p size, padded with K_PAD_VALUE, as a PNG below @p folder."""
    import cv2  # noqa: PLC0415
    import numpy  # noqa: PLC0415

    pixels = cv2.imread(str(source))
    if pixels is None:
        raise ExportError(f"cannot read the image '{source}'")
    geometry = letterbox_geometry(pixels.shape[0], pixels.shape[1], size)
    resized = cv2.resize(pixels, (geometry.width, geometry.height),
                         interpolation=cv2.INTER_LINEAR)
    fitted = numpy.full((size[0], size[1], 3), K_PAD_VALUE, dtype=numpy.uint8)
    fitted[: geometry.height, : geometry.width] = resized
    target = folder / image_name(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), fitted):
        raise OSError(f"could not write {target}")
    return target


class OnnxRunner:
    """Runs an ONNX file on the CPU by onnxruntime, or by onnx.reference when it is absent."""

    def __init__(self, path: Path) -> None:
        """Opens @p path and records which runtime runs it."""
        import onnx  # noqa: PLC0415

        model = onnx.load(str(path))
        self.input_name = model.graph.input[0].name
        self.output_names = [output.name for output in model.graph.output]
        try:
            import onnxruntime  # noqa: PLC0415

            self._session: Any = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"],
            )
            self.runtime = f"onnxruntime {onnxruntime.__version__}"
        except ImportError:
            from onnx.reference import ReferenceEvaluator  # noqa: PLC0415

            self._session = ReferenceEvaluator(model)
            self.runtime = f"onnx.reference {onnx.__version__}"

    def run(self, batch: Any) -> dict[str, Any]:
        """Returns every output of the file for @p batch, `[1, 3, H, W]` float32, by name."""
        values = self._session.run(self.output_names, {self.input_name: batch})
        return dict(zip(self.output_names, values))


def detect(runner: OnnxRunner, image: Path, size: tuple[int, int], mean: Sequence[float],
           std: Sequence[float], score_threshold: float, iou_threshold: float) -> list[list[float]]:
    """Returns the detection rows of the letterboxed PNG @p image, highest score first."""
    import cv2  # noqa: PLC0415
    import numpy  # noqa: PLC0415

    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB).astype(numpy.float32)
    scaled = (rgb / numpy.float32(255.0) - numpy.array(mean, dtype=numpy.float32)) / numpy.array(
        std, dtype=numpy.float32,
    )
    batch = numpy.ascontiguousarray(scaled.transpose(2, 0, 1)[None], dtype=numpy.float32)
    outputs = runner.run(batch)
    candidates: list[tuple[float, int, list[float]]] = []
    for level, stride in enumerate(K_STRIDES):
        nhwc = {
            role: outputs[port_name(level, role)].transpose(0, 2, 3, 1)
            for role in K_ROLES
        }
        grid = (size[0] // stride, size[1] // stride)
        if nhwc["boxes"].shape[1:3] != grid:
            raise ExportError(f"level {level} is {nhwc['boxes'].shape[1:3]}, not {grid}")
        candidates.extend(decode_level(
            nhwc["boxes"].flatten().tolist(),
            nhwc["objectness"].flatten().tolist(),
            nhwc["classes"].flatten().tolist(),
            grid,
            stride,
            score_threshold,
        ))
    kept = suppress(candidates, iou_threshold)
    return [detection_row(cls, score, box) for score, cls, box in kept]


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Writes a YOLOX ONNX file's detections.")
    parser.add_argument("data_dir", type=Path, help="folder holding the ONNX file")
    parser.add_argument("images", nargs="+", type=Path, help="source images")
    parser.add_argument("--onnx", default=K_DEFAULT_ONNX, help="file name in DATA_DIR")
    parser.add_argument("--size", nargs=2, type=int, default=list(K_DEFAULT_SIZE),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--mean", nargs=3, type=float, default=list(K_DEFAULT_MEAN),
                        metavar=("R", "G", "B"))
    parser.add_argument("--std", nargs=3, type=float, default=list(K_DEFAULT_STD),
                        metavar=("R", "G", "B"))
    parser.add_argument("--score", type=float, default=K_SCORE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=K_SUPPRESSION_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Writes the letterboxed images and the reference file; returns the exit code."""
    arguments = parse_arguments(argv)
    size = (arguments.size[0], arguments.size[1])
    try:
        folder = arguments.data_dir / K_REFERENCE_FOLDER
        runner = OnnxRunner(arguments.data_dir / arguments.onnx)
        check_output_names(runner.output_names)
        entries = []
        for source in arguments.images:
            fitted = write_letterbox(source, folder, size)
            rows = detect(runner, fitted, size, arguments.mean, arguments.std,
                          arguments.score, arguments.iou)
            entries.append({"file": fitted.relative_to(folder).as_posix(), "detections": rows})
        document = reference_document(arguments.onnx, runner.runtime, size, arguments.mean,
                                      arguments.std, arguments.score, arguments.iou, entries)
        target = folder / K_REFERENCE_NAME
        target.write_text(json.dumps(document, indent=1), encoding="utf-8")
        print(f"{target}: {sum(len(e['detections']) for e in entries)} detections")
    except (ExportError, ImportError, OSError, ValueError) as error:
        print(f"export_yolox_onnx_reference: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
