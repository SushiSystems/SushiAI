# export_ultralytics_detections.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Writes Ultralytics' detections of released checkpoints on coco128 as reference JSON files.

Usage: python tools/reference/export_ultralytics_detections.py DATA_DIR COCO_IMAGES
       [--count N] [--models yolo26n yolov8n ...]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

K_FORMAT_VERSION = 1
K_IMAGE_SIDE = 640
K_SCORE_THRESHOLD = 0.25
K_SUPPRESSION_THRESHOLD = 0.7
K_DEFAULT_COUNT = 16
K_DEFAULT_MODELS = ("yolo26n", "yolov8n", "yolo11n", "yolov5nu")
K_REFERENCE_FOLDER = "references"
K_IMAGE_FOLDER = "images"


def image_name(source: Path) -> str:
    """Returns the name, below the reference folder, of the square copy of @p source.

    Args:
        source: A coco128 image.

    Returns:
        ``images/<stem>.png``; PNG keeps the resized pixels exactly.
    """
    return f"{K_IMAGE_FOLDER}/{source.stem}.png"


def detection_row(cls: int, score: float, box: Sequence[float]) -> list[float]:
    """Returns one detection as the reference file holds it: class, score, x1, y1, x2, y2.

    Args:
        cls: The class index.
        score: The class probability.
        box: Pixel corners x1, y1, x2, y2, clipped to the image as Ultralytics clips them.

    Returns:
        Six numbers, the class first.
    """
    return [float(cls), float(score), *(float(v) for v in box)]


def reference_document(model: str, version: str, images: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns the reference file of one model.

    Args:
        model: The checkpoint's stem, such as ``yolov8n``.
        version: The installed Ultralytics version.
        images: One entry per image: its file and its detections, highest score first.

    Returns:
        The document ``write_reference`` writes as JSON.
    """
    return {
        "format_version": K_FORMAT_VERSION,
        "ultralytics": version,
        "checkpoint": f"{model}.pt",
        "side": K_IMAGE_SIDE,
        "score_threshold": K_SCORE_THRESHOLD,
        "suppression_threshold": K_SUPPRESSION_THRESHOLD,
        "images": images,
    }


def write_squares(sources: Sequence[Path], folder: Path) -> list[Path]:
    """Writes each source resized to K_IMAGE_SIDE square, so no side letterboxes it.

    Args:
        sources: The coco128 images, in order.
        folder: The reference folder.

    Returns:
        The written files, in the order of @p sources.
    """
    import cv2

    (folder / K_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)
    written = []
    for source in sources:
        target = folder / image_name(source)
        pixels = cv2.resize(cv2.imread(str(source)), (K_IMAGE_SIDE, K_IMAGE_SIDE),
                            interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(target), pixels):
            raise OSError(f"could not write {target}")
        written.append(target)
    return written


def predict(checkpoint: Path, images: Sequence[Path], folder: Path) -> list[dict[str, Any]]:
    """Returns Ultralytics' detections of @p checkpoint on @p images, on the CPU in float32.

    Args:
        checkpoint: The released ``.pt`` file.
        images: The square images.
        folder: The reference folder the image names are relative to.

    Returns:
        One entry per image, as ``reference_document`` takes them.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    entries = []
    for image in images:
        result = model.predict(str(image), imgsz=K_IMAGE_SIDE, conf=K_SCORE_THRESHOLD,
                               iou=K_SUPPRESSION_THRESHOLD, device="cpu",
                               verbose=False)[0]
        rows = [detection_row(int(c), float(s), [float(v) for v in b])
                for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
        rows.sort(key=lambda row: -row[1])
        entries.append({"file": image.relative_to(folder).as_posix(), "detections": rows})
    return entries


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Writes Ultralytics' coco128 detections.")
    parser.add_argument("data_dir", type=Path, help="folder holding <model>.pt")
    parser.add_argument("coco_images", type=Path, help="coco128 images/train2017 folder")
    parser.add_argument("--count", type=int, default=K_DEFAULT_COUNT, help="images, by name")
    parser.add_argument("--models", nargs="+", default=list(K_DEFAULT_MODELS), help="stems")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Writes the square images and one reference file per model; returns the exit code."""
    arguments = parse_arguments(argv)
    try:
        folder = arguments.data_dir / K_REFERENCE_FOLDER
        sources = sorted(arguments.coco_images.glob("*.jpg"))[: arguments.count]
        if len(sources) != arguments.count:
            raise FileNotFoundError(f"{arguments.coco_images} holds fewer than "
                                    f"{arguments.count} .jpg images")
        images = write_squares(sources, folder)
        version = importlib.metadata.version("ultralytics")
        for model in arguments.models:
            entries = predict(arguments.data_dir / f"{model}.pt", images, folder)
            target = folder / f"{model}.json"
            target.write_text(json.dumps(reference_document(model, version, entries), indent=1),
                              encoding="utf-8")
            print(f"{target}: {sum(len(e['detections']) for e in entries)} detections")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"export_ultralytics_detections: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
