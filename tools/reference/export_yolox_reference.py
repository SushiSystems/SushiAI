# export_yolox_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Exports the head outputs of ByteTrack's YOLOX-X on one image as a SACP reference file.

The script runs YOLOX under PyTorch, outside this repository, and writes the network input and
the raw outputs of the three levels, before any sigmoid or grid decoding.
Usage: see tools/reference/README.md, "export_yolox_reference.py".
"""

from __future__ import annotations

import argparse
import inspect
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from export_yolo26_reference import SacpRecord, encode_sacp
except ImportError:  # imported as reference.export_yolox_reference, with tools/ on the path
    from reference.export_yolo26_reference import SacpRecord, encode_sacp

K_DEPTH = 1.33
K_WIDTH = 1.25
K_CLASS_COUNT = 1
K_IN_CHANNELS = (256, 512, 1024)
K_STRIDES = (8, 16, 32)
K_DEFAULT_SIZE = (800, 1440)
K_PAD_VALUE = 114
K_ROLES = ("classes", "boxes", "objectness")
K_ROLE_CHANNELS = {"classes": K_CLASS_COUNT, "boxes": 4, "objectness": 1}
K_PREDICTORS = {"classes": "cls_preds", "boxes": "reg_preds", "objectness": "obj_preds"}
K_INPUT_NAME = "input"
K_REFERENCE_FILE = "yolox_x_reference.sacp"
K_CONTEXT_FILE = "yolox_x_reference.json"
K_FORMAT_VERSION = 1


class ExportError(Exception):
    """Reports a failure of the YOLOX reference export."""


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    """Holds how YOLOX fits an image into the network input: top left, padded right and bottom."""

    ratio: float
    height: int
    width: int


def port_name(level: int, role: str) -> str:
    """Returns the name of one head output as the model file publishes it: `level<l>.<role>`.

    Args:
        level: The level, 0 for the finest grid.
        role: One of `classes`, `boxes` and `objectness`.

    Raises:
        ExportError: The level or the role is not one a YOLOX head has.
    """
    if not 0 <= level < len(K_STRIDES):
        raise ExportError(f"level {level} is not one of {len(K_STRIDES)} levels")
    if role not in K_ROLES:
        raise ExportError(f"'{role}' is not one of {K_ROLES}")
    return f"level{level}.{role}"


def letterbox_geometry(
    image_height: int,
    image_width: int,
    size: tuple[int, int],
) -> LetterboxGeometry:
    """Returns the size YOLOX's `preproc` resizes an image to before it pads.

    The ratio is the smaller of the two size-to-image ratios, and the resized sides truncate to
    integers, as `int(image.shape[1] * r)` does in YOLOX.

    Args:
        image_height: Rows of the source image.
        image_width: Columns of the source image.
        size: The network input as (height, width).

    Raises:
        ExportError: An extent is not positive.
    """
    if min(image_height, image_width, *size) <= 0:
        raise ExportError(
            f"extents must be positive: image {image_height}x{image_width}, size {size}",
        )
    ratio = min(size[0] / image_height, size[1] / image_width)
    return LetterboxGeometry(ratio, int(image_height * ratio), int(image_width * ratio))


def nchw_to_nhwc(values: Sequence[float], shape: tuple[int, int, int, int]) -> list[float]:
    """Returns the values of an NCHW array laid out NHWC.

    Args:
        values: The array flattened in NCHW order.
        shape: The extents (N, C, H, W).

    Raises:
        ExportError: The values do not fill the shape.
    """
    batch, channels, height, width = shape
    if len(values) != batch * channels * height * width:
        raise ExportError(f"{len(values)} values do not fill the shape {shape}")
    out: list[float] = []
    for n in range(batch):
        for h in range(height):
            for w in range(width):
                for c in range(channels):
                    out.append(values[((n * channels + c) * height + h) * width + w])
    return out


def float32_record(name: str, shape: Sequence[int], values: Sequence[float]) -> SacpRecord:
    """Returns a float32 record of the values in the order given."""
    return SacpRecord(name, "float32", tuple(shape), struct.pack(f"<{len(values)}f", *values))


def head_records(
    levels: Sequence[Mapping[str, tuple[tuple[int, ...], Sequence[float]]]],
) -> list[SacpRecord]:
    """Returns the nine head records, level by level, each laid out NHWC.

    Args:
        levels: One mapping per level, finest first, from a role to its NCHW shape and flat values.

    Raises:
        ExportError: The level count is not three, a role is missing or has the wrong channel
            count, or a level's three outputs sit on different grids.
    """
    if len(levels) != len(K_STRIDES):
        raise ExportError(f"{len(levels)} levels; a YOLOX-X head has {len(K_STRIDES)}")
    records: list[SacpRecord] = []
    for level, outputs in enumerate(levels):
        grids: set[tuple[int, int]] = set()
        for role in K_ROLES:
            if role not in outputs:
                raise ExportError(f"level {level} has no '{role}' output")
            shape, values = outputs[role]
            if len(shape) != 4 or shape[1] != K_ROLE_CHANNELS[role]:
                raise ExportError(
                    f"level {level} '{role}' has shape {shape}; it needs "
                    f"{K_ROLE_CHANNELS[role]} channels",
                )
            grids.add((shape[2], shape[3]))
            nhwc = (shape[0], shape[2], shape[3], shape[1])
            records.append(
                float32_record(port_name(level, role), nhwc, nchw_to_nhwc(values, shape)),
            )
        if len(grids) != 1:
            raise ExportError(f"level {level} outputs sit on different grids: {sorted(grids)}")
    return records


def input_record(shape_nchw: tuple[int, int, int, int], values: Sequence[float]) -> SacpRecord:
    """Returns the network input as an NHWC record named `input`."""
    batch, channels, height, width = shape_nchw
    return float32_record(
        K_INPUT_NAME,
        (batch, height, width, channels),
        nchw_to_nhwc(values, shape_nchw),
    )


def reference_context(
    checkpoint: Path,
    image: Path,
    size: tuple[int, int],
    geometry: LetterboxGeometry,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    yolox_version: str,
) -> dict[str, Any]:
    """Returns the facts that describe how the reference tensors were produced."""
    return {
        "format_version": K_FORMAT_VERSION,
        "checkpoint": checkpoint.name,
        "image": image.name,
        "input_size": list(size),
        "resized": [geometry.height, geometry.width],
        "ratio": geometry.ratio,
        "pad_value": K_PAD_VALUE,
        "mean": None if mean is None else list(mean),
        "std": None if std is None else list(std),
        "yolox": yolox_version,
        "outputs": [port_name(level, role) for level in range(len(K_STRIDES)) for role in K_ROLES],
        "layout": "NHWC, raw logits and offsets, no sigmoid and no grid decoding",
    }


def write_reference(
    output_dir: Path,
    records: Sequence[SacpRecord],
    context: Mapping[str, Any],
) -> list[Path]:
    """Writes the SACP file and its context file into @p output_dir and returns both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = output_dir / K_REFERENCE_FILE
    description = output_dir / K_CONTEXT_FILE
    reference.write_bytes(encode_sacp(records))
    description.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    return [reference, description]


def build_model() -> Any:
    """Returns YOLOX-X with one class, as ByteTrack's `yolox_x_mix_det` experiment builds it."""
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # noqa: PLC0415

    backbone = YOLOPAFPN(K_DEPTH, K_WIDTH, in_channels=list(K_IN_CHANNELS))
    head = YOLOXHead(K_CLASS_COUNT, K_WIDTH, in_channels=list(K_IN_CHANNELS))
    return YOLOX(backbone, head)


def load_checkpoint(model: Any, checkpoint: Path) -> None:
    """Loads the `model` state dictionary of a YOLOX checkpoint into @p model."""
    import torch  # noqa: PLC0415

    state = torch.load(checkpoint, map_location="cpu")
    if "model" not in state:
        raise ExportError(f"'{checkpoint}' holds no 'model' state dictionary")
    model.load_state_dict(state["model"])


def preprocess(
    image_path: Path,
    size: tuple[int, int],
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> tuple[Any, LetterboxGeometry]:
    """Returns the network input and the fit, from YOLOX's own `preproc`.

    The mean and the standard deviation are passed only when the installed `preproc` takes them,
    as ByteTrack's does; a later YOLOX takes neither.
    """
    import cv2  # noqa: PLC0415
    from yolox.data.data_augment import preproc  # noqa: PLC0415

    image = cv2.imread(str(image_path))
    if image is None:
        raise ExportError(f"cannot read the image '{image_path}'")
    parameters = inspect.signature(preproc).parameters
    arguments: dict[str, Any] = {}
    if "mean" in parameters:
        arguments = {"mean": mean, "std": std}
    elif mean is not None or std is not None:
        raise ExportError("the installed yolox preproc takes no mean or std")
    prepared = preproc(image, size, **arguments)
    array = prepared[0] if isinstance(prepared, tuple) else prepared
    return array, letterbox_geometry(image.shape[0], image.shape[1], size)


def capture_levels(model: Any, batch: Any) -> list[dict[str, tuple[tuple[int, ...], list[float]]]]:
    """Runs @p model on @p batch in eval mode and returns the raw output of each predictor."""
    import torch  # noqa: PLC0415

    captured: dict[tuple[int, str], Any] = {}
    handles = []
    for role, attribute in K_PREDICTORS.items():
        for level, module in enumerate(getattr(model.head, attribute)):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _i, output, key=(level, role): captured.__setitem__(key, output),
                ),
            )
    model.eval()
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return [
        {
            role: (
                tuple(captured[(level, role)].shape),
                captured[(level, role)].flatten().tolist(),
            )
            for role in K_ROLES
        }
        for level in range(len(K_STRIDES))
    ]


def run(
    checkpoint: Path,
    image: Path,
    output_dir: Path,
    size: tuple[int, int],
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> list[Path]:
    """Exports the reference of @p image under @p checkpoint and returns the files written."""
    import torch  # noqa: PLC0415
    import yolox  # noqa: PLC0415

    model = build_model()
    load_checkpoint(model, checkpoint)
    array, geometry = preprocess(image, size, mean, std)
    batch = torch.from_numpy(array).unsqueeze(0).float()
    records = [input_record(tuple(batch.shape), batch.flatten().tolist())]
    records.extend(head_records(capture_levels(model, batch)))
    context = reference_context(
        checkpoint, image, size, geometry, mean, std, getattr(yolox, "__version__", "unknown"),
    )
    return write_reference(output_dir, records, context)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Writes YOLOX-X's head outputs on one image.")
    parser.add_argument("checkpoint", type=Path, help="ByteTrack's mot17x.pth.tar")
    parser.add_argument("image", type=Path, help="one MOT17 frame")
    parser.add_argument("output_dir", type=Path, help="folder that receives the two files")
    parser.add_argument("--size", nargs=2, type=int, default=list(K_DEFAULT_SIZE),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--mean", nargs=3, type=float, default=None, metavar=("R", "G", "B"))
    parser.add_argument("--std", nargs=3, type=float, default=None, metavar=("R", "G", "B"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the export and returns the exit code: 0 on success, 1 with one line on stderr."""
    arguments = parse_arguments(argv)
    try:
        written = run(
            arguments.checkpoint,
            arguments.image,
            arguments.output_dir,
            (arguments.size[0], arguments.size[1]),
            arguments.mean,
            arguments.std,
        )
    except (ExportError, ImportError, OSError) as error:
        print(f"export_yolox_reference: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
