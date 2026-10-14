# export_yolo26_reference.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Exports YOLO26n reference tensors from the pinned Ultralytics commit as SACP files.

Usage: python tools/reference/export_yolo26_reference.py OUTPUT_DIR [--seed N]
       [--bn-state {perturbed,initial}] [--external-weights DIR]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

K_ULTRALYTICS_COMMIT = "ce1399439b93c490b648b55dedd0a6334f0199da"
K_MODEL_YAML = "yolo26n.yaml"
K_MODEL_SCALE = "n"
K_CLASS_COUNT = 80
K_IMAGE_SIDE = 640
K_THREAD_COUNT = 2
K_DEFAULT_SEED = 0
K_FIXTURE_CEILING_BYTES = 64 * 1024 * 1024
K_MANIFEST_NAME = "MANIFEST.json"
K_MANIFEST_FORMAT_VERSION = 1
K_WEIGHTS_FILE = "weights.sushi"
K_REGISTRATION_FIXTURES = "vision/registration/fixtures/yolo26"
K_DETECT_FIXTURES = "vision/detect/fixtures/yolo26"
K_BATCHNORM_FIXTURES = "layers/nn/fixtures/batch_norm2d"
K_DESTINATIONS = {
    K_MANIFEST_NAME: (K_REGISTRATION_FIXTURES,),
    K_WEIGHTS_FILE: (K_REGISTRATION_FIXTURES,),
    "input.sacp": (K_REGISTRATION_FIXTURES,),
    "forward.sacp": (K_DETECT_FIXTURES, K_REGISTRATION_FIXTURES),
    "loss.sacp": (K_DETECT_FIXTURES,),
    "batchnorm.sacp": (K_BATCHNORM_FIXTURES,),
}
K_BRANCHES = ("one2many", "one2one")
K_GAIN_NAMES = ("box", "cls", "dfl")
K_ASSIGNER_INPUTS = ("scores", "boxes")
K_VALIDATION_CONFIDENCE = 0.001
K_SUPPRESSION_CONFIDENCE = 1.8e-4
K_SUPPRESSION_TENSOR = "one2many.suppressed.detections"
K_FEATURE_ROWS = (9, 10, 16, 19, 21, 22)
K_FEATURE_PREFIX = "features"

K_SACP_MAGIC = 0x50434153
K_SACP_VERSION = 1
K_SACP_MAX_TENSOR_COUNT = 4096
K_SACP_MAX_NAME_LENGTH = 256
K_SACP_MAX_RANK = 6
K_SACP_DTYPES = {
    "half": (0, 2),
    "float32": (1, 4),
    "float64": (2, 8),
    "complex32": (3, 8),
    "complex64": (4, 16),
    "int32": (5, 4),
    "int64": (6, 8),
}
K_WRITTEN_DTYPES = {"float32": "float32", "int32": "int32", "int64": "int64", "bool": "int32"}

K_LAYOUT_CONV = "conv_weight"
K_LAYOUT_DEPTHWISE = "depthwise_weight"
K_LAYOUT_VECTOR = "channel_vector"
K_LAYOUT_UNSPECIFIED = "unspecified"
K_LAYOUT_NHWC = "nchw_to_nhwc"
K_LAYOUT_ANCHOR_ROWS = "anchor_rows"
K_LAYOUT_AS_IS = "as_is"
K_VECTOR_KINDS = frozenset(
    {
        ("Conv2d", "bias"),
        ("BatchNorm2d", "weight"),
        ("BatchNorm2d", "bias"),
        ("BatchNorm2d", "running_mean"),
        ("BatchNorm2d", "running_var"),
    },
)
K_SKIPPED_LEAVES = frozenset({"num_batches_tracked"})

K_BN_PERTURBATION = {
    "weight": (0.5, 1.5),
    "bias": (-0.25, 0.25),
    "running_mean": (-0.25, 0.25),
    "running_var": (0.5, 1.5),
}
K_GROUND_TRUTH = (
    (0.0, 0.50, 0.50, 0.40, 0.60),
    (17.0, 0.25, 0.30, 0.20, 0.15),
    (41.0, 0.60, 0.55, 0.30, 0.30),
    (79.0, 0.80, 0.20, 0.015, 0.0125),
)
K_GRADIENT_TENSORS = (
    "model.0.conv.weight",
    "model.23.cv2.0.2.bias",
    "model.23.one2one_cv3.0.2.bias",
)
K_GRADIENT_SOURCE = "torch.autograd, backward of total_loss"
K_STATISTICS_MODULE = "model.0.bn"
K_BATCHNORM_FILE = "batchnorm.sacp"
K_BATCHNORM_PREFIX = "batchnorm_factor"
K_BATCHNORM_UNIT_CASE = "unit"
K_BATCHNORM_UNIT_SHAPE = (2, 3, 2, 2)
K_BATCHNORM_MODEL_MODULE = "model.9.cv1.bn"
K_BATCHNORM_BIASED = "biased"
K_BATCHNORM_UNBIASED = "unbiased"
K_BATCHNORM_CANDIDATES = (K_BATCHNORM_BIASED, K_BATCHNORM_UNBIASED)
K_BATCHNORM_MATCH_ULPS = 4
K_BATCHNORM_OBSERVED = {
    "input": K_LAYOUT_NHWC,
    "weight": K_LAYOUT_VECTOR,
    "bias": K_LAYOUT_VECTOR,
    "before.running_mean": K_LAYOUT_VECTOR,
    "before.running_var": K_LAYOUT_VECTOR,
    "output": K_LAYOUT_NHWC,
    "after.running_mean": K_LAYOUT_VECTOR,
    "after.running_var": K_LAYOUT_VECTOR,
}
K_FOREIGN_ERRORS = (
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class ExportError(Exception):
    """Reports a failure of the reference export."""


class SacpFormatError(ExportError):
    """Reports a record the SACP format cannot hold."""


class NetworkAccessError(ExportError):
    """Reports that the reference code tried to reach the network."""


class PinMismatchError(ExportError):
    """Reports an installed Ultralytics that is not the pinned commit."""


class ReferenceMismatchError(ExportError):
    """Reports an exported tensor that differs from what the reference returned."""


class FixtureTooLargeError(ExportError):
    """Reports a fixture set above the repository ceiling."""


class BlockedConnectionError(OSError):
    """Reports one network call the guard refused."""


@dataclass(frozen=True, slots=True)
class SacpRecord:
    """Holds one tensor as the SACP format stores it."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True, slots=True)
class ExportedTensor:
    """Holds one written tensor and the facts the manifest records about it."""

    record: SacpRecord
    source: str
    layout: str
    original_shape: tuple[int, ...]
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True, slots=True)
class ReferenceFile:
    """Holds the tensors of one SACP file and the conditions they were produced under."""

    name: str
    tensors: tuple[ExportedTensor, ...]
    context: dict[str, Any]


def encode_sacp(records: Sequence[SacpRecord]) -> bytes:
    """Returns the bytes of a SACP file holding the records in order.

    Raises:
        SacpFormatError: A record breaks a bound of the format, or a name repeats.
    """
    if len(records) > K_SACP_MAX_TENSOR_COUNT:
        raise SacpFormatError(f"{len(records)} tensors; a file holds {K_SACP_MAX_TENSOR_COUNT}")
    chunks = [struct.pack("<III", K_SACP_MAGIC, K_SACP_VERSION, len(records))]
    seen: set[str] = set()
    for record in records:
        if record.name in seen:
            raise SacpFormatError(f"'{record.name}' is written twice")
        seen.add(record.name)
        chunks.append(_encode_record(record))
    return b"".join(chunks)


def _encode_record(record: SacpRecord) -> bytes:
    """Returns one record's header and payload as bytes."""
    name = record.name.encode("utf-8")
    if len(name) > K_SACP_MAX_NAME_LENGTH:
        raise SacpFormatError(f"'{record.name[:32]}...' exceeds {K_SACP_MAX_NAME_LENGTH} bytes")
    if record.dtype not in K_SACP_DTYPES:
        raise SacpFormatError(f"'{record.name}' has dtype '{record.dtype}', which SACP lacks")
    if len(record.shape) > K_SACP_MAX_RANK:
        raise SacpFormatError(f"'{record.name}' has rank {len(record.shape)} > {K_SACP_MAX_RANK}")
    if any(extent <= 0 for extent in record.shape):
        raise SacpFormatError(f"'{record.name}' has a non-positive extent in {record.shape}")
    code, size = K_SACP_DTYPES[record.dtype]
    expected = size * functools.reduce(lambda count, extent: count * extent, record.shape, 1)
    if len(record.payload) != expected:
        raise SacpFormatError(
            f"'{record.name}' carries {len(record.payload)} payload bytes; "
            f"its shape needs {expected}",
        )
    header = struct.pack("<I", len(name)) + name + struct.pack("<II", code, len(record.shape))
    return header + struct.pack(f"<{len(record.shape)}q", *record.shape) + record.payload


def classify_layout(module_kind: str, leaf: str, groups: int, shape: tuple[int, ...]) -> str:
    """Returns the import layout the YOLO26 SPEC names for one state-dict tensor.

    Args:
        module_kind: Class name of the module that owns the tensor.
        leaf: Last component of the state-dict key.
        groups: Group count of the owner; 1 for a module without one.
        shape: Shape of the tensor in the reference layout.
    """
    if module_kind == "Conv2d" and leaf == "weight" and len(shape) == 4:
        if groups == 1:
            return K_LAYOUT_CONV
        if groups == shape[0] and shape[1] == 1:
            return K_LAYOUT_DEPTHWISE
        return K_LAYOUT_UNSPECIFIED
    if len(shape) == 1 and (module_kind, leaf) in K_VECTOR_KINDS:
        return K_LAYOUT_VECTOR
    return K_LAYOUT_UNSPECIFIED


def name_feature_map(row: int, head_rows: Sequence[int]) -> str:
    """Returns the tensor name of one model row's output.

    Args:
        row: Index of the row in the model yaml.
        head_rows: Rows the detect head reads, in level order.
    """
    if row in head_rows:
        return f"{K_FEATURE_PREFIX}.level{list(head_rows).index(row)}"
    return f"{K_FEATURE_PREFIX}.row{row}"


def encode_files(files: Sequence[ReferenceFile]) -> dict[str, bytes]:
    """Returns the SACP bytes of each reference file by file name."""
    return {
        reference.name: encode_sacp([tensor.record for tensor in reference.tensors])
        for reference in files
    }


def build_manifest(
    run_facts: dict[str, Any],
    files: Sequence[ReferenceFile],
    payloads: dict[str, bytes],
    external: bool,
) -> str:
    """Returns the manifest text for the files, with each file's digest and size."""
    entries = []
    for reference in files:
        data = payloads[reference.name]
        is_external = external and reference.name == K_WEIGHTS_FILE
        entries.append(
            {
                "name": reference.name,
                "location": "external" if is_external else "fixture_folder",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "context": reference.context,
                "tensors": [_describe_tensor(tensor) for tensor in reference.tensors],
            },
        )
    manifest = {"format_version": K_MANIFEST_FORMAT_VERSION, **run_facts, "files": entries}
    return json.dumps(manifest, indent=2, allow_nan=False) + "\n"


def _describe_tensor(tensor: ExportedTensor) -> dict[str, Any]:
    """Returns the manifest entry of one tensor."""
    return {
        "name": tensor.record.name,
        "dtype": tensor.record.dtype,
        "shape": list(tensor.record.shape),
        "original_shape": list(tensor.original_shape),
        "layout": tensor.layout,
        "source": tensor.source,
        "min": tensor.minimum,
        "max": tensor.maximum,
        "mean": tensor.mean,
    }


def write_reference_files(
    payloads: dict[str, bytes],
    manifest: str,
    tests_dir: Path,
    external_weights: Path | None,
) -> dict[str, int]:
    """Writes each file to the fixture folders K_DESTINATIONS names below tests_dir.

    Returns each written size by file name.

    Raises:
        FixtureTooLargeError: The files of one fixture folder would exceed the fixture ceiling.
    """
    contents = {**payloads, K_MANIFEST_NAME: manifest.encode("utf-8")}
    targets = [
        (name, tests_dir / folder / name) for name in contents for folder in K_DESTINATIONS[name]
    ]
    if external_weights is not None:
        targets = [(name, path) for name, path in targets if name != K_WEIGHTS_FILE]
        targets.append((K_WEIGHTS_FILE, external_weights / K_WEIGHTS_FILE))
    for folder in {tests_dir / folder for folders in K_DESTINATIONS.values() for folder in folders}:
        in_folder = sum(len(contents[name]) for name, path in targets if path.parent == folder)
        if in_folder > K_FIXTURE_CEILING_BYTES:
            raise FixtureTooLargeError(
                f"{in_folder} bytes in {folder} exceed the {K_FIXTURE_CEILING_BYTES} byte "
                "ceiling; pass --external-weights DIR to keep weights.sushi outside it",
            )
    for name, path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[name])
    return {name: len(data) for name, data in contents.items()}


def guard_network() -> list[str]:
    """Blocks every socket connection and name lookup and returns the list of refused calls."""
    attempts: list[str] = []

    def refuse(label: str, *arguments: Any, **_keywords: Any) -> None:
        target = arguments[1] if label == "connect" else arguments[0]
        attempts.append(f"{label} {target!r}")
        raise BlockedConnectionError(f"network access is blocked: {label} {target!r}")

    socket.getaddrinfo = functools.partial(refuse, "getaddrinfo")
    socket.create_connection = functools.partial(refuse, "create_connection")
    socket.socket.connect = functools.partialmethod(refuse, "connect")
    socket.socket.connect_ex = functools.partialmethod(refuse, "connect")
    return attempts


def configure_environment() -> None:
    """Sets the variables that keep Ultralytics offline and both libraries on two threads."""
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "0"
    os.environ["YOLO_VERBOSE"] = "0"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(K_THREAD_COUNT)


def load_torch(seed: int) -> None:
    """Imports torch, caps its threads, seeds it and turns deterministic algorithms on."""
    import torch

    torch.set_num_threads(K_THREAD_COUNT)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)


def read_installed_commit() -> str:
    """Returns the commit the installed Ultralytics distribution was built from.

    Raises:
        PinMismatchError: The distribution records no commit, or not the pinned one.
    """
    text = importlib.metadata.distribution("ultralytics").read_text("direct_url.json")
    commit = json.loads(text)["vcs_info"]["commit_id"] if text else ""
    if commit != K_ULTRALYTICS_COMMIT:
        raise PinMismatchError(f"Ultralytics is at '{commit}'; the pin is {K_ULTRALYTICS_COMMIT}")
    return commit


def describe_source(target: Callable[..., Any]) -> str:
    """Returns the Ultralytics file, first line and qualified name of a callable."""
    parts = Path(inspect.getsourcefile(target)).parts
    start = len(parts) - 1 - parts[::-1].index("ultralytics")
    line = inspect.getsourcelines(target)[1]
    return f"{'/'.join(parts[start:])}:{line} ({target.__qualname__})"


def to_exported(
    name: str,
    tensor: torch.Tensor,
    source: str,
    layout: str,
    original_shape: Sequence[int],
) -> ExportedTensor:
    """Returns the tensor as little-endian bytes with its statistics taken from those bytes.

    Raises:
        ReferenceMismatchError: The dtype has no SACP counterpart, or a value is not finite.
    """
    import numpy

    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.name not in K_WRITTEN_DTYPES:
        raise ReferenceMismatchError(f"'{name}' is {array.dtype.name}, which SACP cannot hold")
    dtype = K_WRITTEN_DTYPES[array.dtype.name]
    little_endian = numpy.dtype(dtype).newbyteorder("<")
    array = numpy.ascontiguousarray(array.reshape(array.shape or (1,)), dtype=little_endian)
    if not numpy.isfinite(array).all():
        raise ReferenceMismatchError(f"'{name}' holds a value that is not finite")
    record = SacpRecord(name, dtype, tuple(int(extent) for extent in array.shape), array.tobytes())
    return ExportedTensor(
        record=record,
        source=source,
        layout=layout,
        original_shape=tuple(int(extent) for extent in original_shape),
        minimum=float(array.min()),
        maximum=float(array.max()),
        mean=float(array.astype(numpy.float64).mean()),
    )


def apply_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Returns the tensor permuted from the reference layout into the layout SushiAI imports."""
    if layout in (K_LAYOUT_CONV, K_LAYOUT_NHWC):
        return tensor.permute(0, 2, 3, 1)
    if layout == K_LAYOUT_DEPTHWISE:
        return tensor.squeeze(1).permute(1, 2, 0)
    if layout == K_LAYOUT_ANCHOR_ROWS:
        return tensor.permute(0, 2, 1)
    return tensor


def export_tensor(
    name: str,
    tensor: torch.Tensor,
    source: str,
    layout: str = K_LAYOUT_AS_IS,
) -> ExportedTensor:
    """Returns a reference-layout tensor permuted into the named layout and ready to write."""
    return to_exported(name, apply_layout(tensor, layout), source, layout, tensor.shape)


def find_layout(model: torch.nn.Module, name: str, shape: Sequence[int]) -> str:
    """Returns the import layout of the state-dict entry or gradient with the given name."""
    owner_name, _, leaf = name.rpartition(".")
    owner = model.get_submodule(owner_name)
    return classify_layout(type(owner).__name__, leaf, getattr(owner, "groups", 1), tuple(shape))


def build_model(seed: int) -> torch.nn.Module:
    """Returns YOLO26n built from its yaml with weights drawn from the seed.

    Raises:
        ReferenceMismatchError: The yaml resolved to another scale or class count.
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    torch.manual_seed(seed)
    model = DetectionModel(K_MODEL_YAML, ch=3, nc=K_CLASS_COUNT, verbose=False)
    model.args = DEFAULT_CFG
    if model.yaml["scale"] != K_MODEL_SCALE or model.model[-1].nc != K_CLASS_COUNT:
        raise ReferenceMismatchError(
            f"the yaml built scale '{model.yaml['scale']}' with {model.model[-1].nc} classes",
        )
    return model


def perturb_normalization(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Overwrites every BatchNorm weight, bias and running statistic with uniform draws."""
    import torch

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                continue
            for attribute, (low, high) in K_BN_PERTURBATION.items():
                target = getattr(module, attribute)
                target.copy_(torch.rand(target.shape, generator=generator) * (high - low) + low)


def make_input(generator: torch.Generator) -> torch.Tensor:
    """Returns one NHWC image batch of uniform draws in [0, 1)."""
    import torch

    return torch.rand((1, K_IMAGE_SIDE, K_IMAGE_SIDE, 3), generator=generator, dtype=torch.float32)


def make_training_batch() -> dict[str, torch.Tensor]:
    """Returns the ground truth of the one training image in the form the reference loss takes."""
    import torch

    rows = torch.tensor(K_GROUND_TRUTH, dtype=torch.float32)
    return {
        "batch_idx": torch.zeros(rows.shape[0], dtype=torch.float32),
        "cls": rows[:, :1].clone(),
        "bboxes": rows[:, 1:].clone(),
    }


def export_input(image_nhwc: torch.Tensor) -> ReferenceFile:
    """Returns the input file: the image in NHWC, which torch receives permuted to NCHW."""
    tensor = export_tensor("image", image_nhwc, f"{Path(__file__).name} (make_input)")
    context = {
        "value_range": "uniform draws in [0, 1), not a photograph",
        "fed_to_torch_as": "image.permute(0, 3, 1, 2), shape [1, 3, 640, 640]",
    }
    return ReferenceFile("input.sacp", (tensor,), context)


def export_weights(model: torch.nn.Module, bn_state: str) -> ReferenceFile:
    """Returns the weights file: every floating state-dict entry in the layout SushiAI imports."""
    from ultralytics.nn.tasks import DetectionModel

    source = describe_source(DetectionModel.__init__)
    tensors: list[ExportedTensor] = []
    skipped: list[str] = []
    for name, tensor in model.state_dict().items():
        if name.rpartition(".")[2] in K_SKIPPED_LEAVES:
            skipped.append(name)
            continue
        tensors.append(export_tensor(name, tensor, source, find_layout(model, name, tensor.shape)))
    normalization = next(module for module in model.modules() if hasattr(module, "running_var"))
    context = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_entries_written": len(tensors),
        "state_entries_skipped": skipped,
        "unspecified_layout": [t.record.name for t in tensors if t.layout == K_LAYOUT_UNSPECIFIED],
        "batch_norm_state": bn_state,
        "batch_norm_perturbation": K_BN_PERTURBATION if bn_state == "perturbed" else {},
        "batch_norm_eps": normalization.eps,
        "batch_norm_momentum": normalization.momentum,
        "strides": [float(stride) for stride in model.stride],
    }
    return ReferenceFile(K_WEIGHTS_FILE, tuple(tensors), context)


def export_forward(model: torch.nn.Module, image_nchw: torch.Tensor) -> ReferenceFile:
    """Returns the forward file from one eval-mode pass: features, raw head outputs and decodes.

    Raises:
        ReferenceMismatchError: The decode exported here differs from what the model returned.
    """
    import torch
    from ultralytics.nn.modules.head import Detect

    head = model.model[-1]
    model.eval()
    with torch.no_grad():
        outputs = watch_rows(model, K_FEATURE_ROWS)
        returned, predictions = model(image_nchw)
        default_end2end = bool(head.end2end)
        decoded = {}
        for branch, end2end in zip(K_BRANCHES, (False, True)):
            head.end2end = end2end
            decoded[branch] = head._inference(predictions[branch])
        detections = head.postprocess(decoded["one2one"].permute(0, 2, 1))
        head.end2end = default_end2end
    expected = detections if default_end2end else decoded["one2many"]
    if not torch.equal(returned, expected):
        raise ReferenceMismatchError("the exported decode differs from the model's own output")

    features = predictions["one2many"]["feats"]
    head_source = describe_source(Detect.forward_head)
    decode_source = describe_source(Detect._inference)
    tensors = export_feature_maps(model, outputs, features, head_source)
    for branch in K_BRANCHES:
        for kind in ("boxes", "scores"):
            for index, level in enumerate(split_levels(predictions[branch][kind], features)):
                name = f"{branch}.level{index}.{kind}"
                tensors.append(export_tensor(name, level, head_source, K_LAYOUT_NHWC))
        shape = decoded[branch].shape
        rows = apply_layout(decoded[branch], K_LAYOUT_ANCHOR_ROWS)
        for kind, part in (("boxes", rows[..., :4]), ("scores", rows[..., 4:])):
            name = f"{branch}.decoded.{kind}"
            tensors.append(to_exported(name, part, decode_source, K_LAYOUT_ANCHOR_ROWS, shape))
    tensors.append(
        export_tensor("one2one.detections", detections, describe_source(Detect.postprocess)),
    )
    suppressed, suppression = export_suppression(decoded["one2many"])
    tensors.append(suppressed)
    context = {
        "mode": "eval",
        "batch_norm_statistics": f"running_mean and running_var as written in {K_WEIGHTS_FILE}",
        "levels": describe_levels(features, model.stride),
        "feature_rows": {name_feature_map(row, head.f): row for row in K_FEATURE_ROWS},
        "anchor_order": "levels in order; inside a level, anchor h * width + w",
        "decoded_box_format": {"one2many": "xywh, pixels", "one2one": "xyxy, pixels"},
        "detection_columns": ["x1", "y1", "x2", "y2", "score", "class_index"],
        "max_detections": int(head.max_det),
        "default_end2end": default_end2end,
        "self_check": "the model's own inference output equals "
        + ("one2one.detections" if default_end2end else "the one2many decode"),
        "validation": describe_validation(decoded["one2many"], detections),
        "suppression": suppression,
    }
    return ReferenceFile("forward.sacp", tuple(tensors), context)


def watch_rows(model: torch.nn.Module, rows: Sequence[int]) -> dict[int, torch.Tensor]:
    """Returns a store the model's next forward pass fills with each listed row's output."""
    store: dict[int, torch.Tensor] = {}
    handles: dict[int, Any] = {}

    def record(row: int, _owner: Any, _inputs: Any, output: torch.Tensor) -> None:
        store[row] = output.detach().clone()
        handles.pop(row).remove()

    for row in rows:
        handles[row] = model.model[row].register_forward_hook(functools.partial(record, row))
    return store


def export_feature_maps(
    model: torch.nn.Module,
    outputs: dict[int, torch.Tensor],
    features: Sequence[torch.Tensor],
    head_source: str,
) -> list[ExportedTensor]:
    """Returns the output of each K_FEATURE_ROWS row in NHWC, in the order of that constant.

    Args:
        outputs: Store of watch_rows, filled by the eval-mode pass.
        features: What the detect head received, in level order.
        head_source: Source recorded for a row the detect head reads.

    Raises:
        ReferenceMismatchError: A row the head reads differs from what the head received.
    """
    import torch

    head_rows = list(model.model[-1].f)
    tensors = []
    for row in K_FEATURE_ROWS:
        output = outputs[row]
        source = head_source
        if row in head_rows:
            if not torch.equal(output, features[head_rows.index(row)]):
                raise ReferenceMismatchError(f"row {row} differs from the head's input")
        else:
            source = describe_source(type(model.model[row]).forward)
        name = name_feature_map(row, head_rows)
        tensors.append(export_tensor(name, output, source, K_LAYOUT_NHWC))
    return tensors


def read_validator_arguments(confidence: float) -> dict[str, Any]:
    """Returns what the reference validator passes to its suppression, at the given confidence."""
    from ultralytics.utils import DEFAULT_CFG

    return {
        "conf_thres": confidence,
        "iou_thres": float(DEFAULT_CFG.iou),
        "multi_label": True,
        "agnostic": bool(DEFAULT_CFG.single_cls or DEFAULT_CFG.agnostic_nms),
        "max_det": int(DEFAULT_CFG.max_det),
    }


def export_suppression(one2many: torch.Tensor) -> tuple[ExportedTensor, dict[str, Any]]:
    """Returns the rows suppression keeps of the first image at a confidence its scores pass.

    Args:
        one2many: Decoded one-to-many output, [B, 4 + nc, A].

    Raises:
        ReferenceMismatchError: No score passes the confidence, so there is no row to write.
    """
    from ultralytics.utils import nms

    arguments = read_validator_arguments(K_SUPPRESSION_CONFIDENCE)
    rows = nms.non_max_suppression(one2many.clone(), nc=0, end2end=False, **arguments)[0]
    if not rows.shape[0]:
        raise ReferenceMismatchError(f"no score passes {K_SUPPRESSION_CONFIDENCE}")
    source = describe_source(nms.non_max_suppression)
    context = {
        "tensor": K_SUPPRESSION_TENSOR,
        "arguments": arguments,
        "candidate_count": int((one2many[0, 4:] > K_SUPPRESSION_CONFIDENCE).sum()),
        "row_count": int(rows.shape[0]),
        "columns": ["x1", "y1", "x2", "y2", "score", "class_index"],
    }
    return export_tensor(K_SUPPRESSION_TENSOR, rows, source), context


def describe_validation(one2many: torch.Tensor, one2one: torch.Tensor) -> dict[str, Any]:
    """Returns the validator's suppression arguments and the rows it keeps of each head.

    Args:
        one2many: Decoded one-to-many output, [B, 4 + nc, A].
        one2one: Top-k detections of the one-to-one head, [B, k, 6].
    """
    from ultralytics.utils import nms

    arguments = read_validator_arguments(K_VALIDATION_CONFIDENCE)
    heads = {"one2many": (one2many, False), "one2one": (one2one, True)}
    rows = {
        branch: nms.non_max_suppression(output.clone(), nc=0, end2end=end2end, **arguments)
        for branch, (output, end2end) in heads.items()
    }
    return {
        "source": describe_source(nms.non_max_suppression),
        "arguments": arguments,
        "suppression": "torchvision.ops.nms" if "torchvision" in sys.modules else "TorchNMS.nms",
        "row_count": {branch: int(kept[0].shape[0]) for branch, kept in rows.items()},
        "largest_score": {
            "one2many": float(one2many[:, 4:].max()),
            "one2one": float(one2one[..., 4].max()),
        },
    }


def split_levels(joined: torch.Tensor, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Returns a [B, C, A] head output cut into one [B, C, H, W] tensor per level."""
    levels = []
    offset = 0
    for feature in features:
        height, width = feature.shape[2:]
        part = joined[:, :, offset : offset + height * width]
        levels.append(part.reshape(joined.shape[0], joined.shape[1], height, width))
        offset += height * width
    return levels


def describe_levels(
    features: Sequence[torch.Tensor],
    strides: torch.Tensor,
) -> list[dict[str, Any]]:
    """Returns each level's stride, grid size, channel count and first anchor index."""
    levels = []
    offset = 0
    for feature, stride in zip(features, strides):
        _, channels, height, width = feature.shape
        levels.append(
            {
                "stride": float(stride),
                "height": int(height),
                "width": int(width),
                "channels": int(channels),
                "anchor_offset": offset,
            },
        )
        offset += int(height * width)
    return levels


def export_loss(
    model: torch.nn.Module,
    image_nchw: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> ReferenceFile:
    """Returns the loss file from one train-mode pass, one loss evaluation and one backward.

    Raises:
        ReferenceMismatchError: The criterion is not the two-branch loss the SPEC describes.
    """
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner

    model.train()
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise ReferenceMismatchError(f"the criterion is {type(criterion).__name__}, not E2ELoss")
    branches = {branch: getattr(criterion, branch) for branch in K_BRANCHES}
    captured: dict[str, tuple[Any, Any]] = {}
    hooks = [
        branches[branch].assigner.register_forward_hook(
            functools.partial(_capture, captured, branch),
        )
        for branch in K_BRANCHES
    ]
    predictions = model(image_nchw)
    items = {branch: branches[branch].loss(predictions[branch], batch)[0] for branch in K_BRANCHES}
    total = (items["one2many"] * criterion.o2m + items["one2one"] * criterion.o2o).sum()
    total.backward()
    for hook in hooks:
        hook.remove()

    loss_source = describe_source(v8DetectionLoss.loss)
    assign_source = describe_source(TaskAlignedAssigner._forward)
    total_source = describe_source(E2ELoss.__call__)
    batch_source = f"{Path(__file__).name} (make_training_batch)"
    tensors = [export_tensor(f"batch.{key}", value, batch_source) for key, value in batch.items()]
    tensors.append(export_tensor("anchors.points", captured[K_BRANCHES[0]][0][2], assign_source))
    foreground = {}
    for branch in K_BRANCHES:
        tensors.append(export_tensor(f"{branch}.loss_items", items[branch], loss_source))
        for key, value in zip(K_ASSIGNER_INPUTS, captured[branch][0]):
            name = f"{branch}.assignment.input.{key}"
            tensors.append(export_tensor(name, value, assign_source))
        labels, boxes, scores, mask, indices = captured[branch][1]
        foreground[branch] = int(mask.sum())
        for key, value in (
            ("foreground_mask", mask),
            ("target_boxes", boxes),
            ("target_scores", scores),
            ("target_indices", indices),
            ("target_labels", labels),
        ):
            tensors.append(export_tensor(f"{branch}.assignment.{key}", value, assign_source))
    weights = total.new_tensor([criterion.o2m, criterion.o2o])
    tensors.append(export_tensor("branch_weights", weights, total_source))
    tensors.append(export_tensor("total_loss", total.reshape(1), total_source))
    tensors.extend(export_gradients(model))
    tensors.extend(export_updated_statistics(model))
    context = {
        "mode": "train",
        "batch_norm_statistics": "batch statistics of the one training image",
        "ground_truth_rows": "batch.cls with batch.bboxes: class, then normalised xywh",
        "loss_item_names": list(criterion.one2many.loss_names),
        "loss_item_scaling": "gain-scaled and multiplied by the batch size, which is 1",
        "gains": {key: float(getattr(criterion.one2many.hyp, key)) for key in K_GAIN_NAMES},
        "total_loss": "sum(one2many.loss_items * branch_weights[0] "
        "+ one2one.loss_items * branch_weights[1])",
        "assigner": {branch: describe_assigner(branches[branch].assigner) for branch in K_BRANCHES},
        "foreground_anchor_count": foreground,
        "assigner_input_format": "scores after sigmoid; boxes xyxy, pixels; both detached",
        "target_box_format": "xyxy, pixels; a row off the foreground holds ground truth 0",
        "anchor_point_format": "x, y of the cell centre, pixels",
        "gradient_of": "total_loss, with no optimizer step taken",
        "statistics_after_forward": f"{K_STATISTICS_MODULE} after the one train-mode pass",
    }
    return ReferenceFile("loss.sacp", tuple(tensors), context)


def _capture(
    store: dict[str, tuple[Any, Any]],
    branch: str,
    _module: Any,
    inputs: Any,
    output: Any,
) -> None:
    """Stores the arguments and the result of one assigner call under its branch name."""
    store[branch] = (inputs, output)


def describe_assigner(assigner: Any) -> dict[str, Any]:
    """Returns the settings the reference built a label assigner with."""
    return {
        "topk": int(assigner.topk),
        "topk2": int(assigner.topk2),
        "alpha": float(assigner.alpha),
        "beta": float(assigner.beta),
        "stride_val": float(assigner.stride_val),
        "eps": float(assigner.eps),
    }


def export_gradients(model: torch.nn.Module) -> list[ExportedTensor]:
    """Returns the gradient of each listed parameter in the layout of the weights file."""
    parameters = dict(model.named_parameters())
    tensors = []
    for name in K_GRADIENT_TENSORS:
        gradient = parameters[name].grad
        layout = find_layout(model, name, gradient.shape)
        tensors.append(export_tensor(f"gradient.{name}", gradient, K_GRADIENT_SOURCE, layout))
    return tensors


def export_updated_statistics(model: torch.nn.Module) -> list[ExportedTensor]:
    """Returns the running statistics of one BatchNorm as the train-mode pass left them."""
    module = model.get_submodule(K_STATISTICS_MODULE)
    source = f"torch.nn.BatchNorm2d, momentum {module.momentum}"
    prefix = f"after_forward.{K_STATISTICS_MODULE}"
    return [
        export_tensor(f"{prefix}.{leaf}", getattr(module, leaf), source, K_LAYOUT_VECTOR)
        for leaf in ("running_mean", "running_var")
    ]


def count_per_channel(shape: Sequence[int]) -> int:
    """Returns how many values of an NCHW batch one channel's statistics are taken over."""
    return int(shape[0]) * int(shape[2]) * int(shape[3])


def scale_batch_variance(candidate: str, count: int) -> float:
    """Returns the factor a candidate applies to the biased batch variance of count values.

    Raises:
        ReferenceMismatchError: The count is below 2, where the unbiased variance is undefined.
    """
    if count < 2:
        raise ReferenceMismatchError(f"a channel holds {count} values; a variance needs 2")
    if candidate == K_BATCHNORM_UNBIASED:
        return count / (count - 1)
    return 1.0


def blend_running_statistic(before: Any, batch: Any, momentum: float, scale: float) -> Any:
    """Returns before * (1 - momentum) + batch * momentum * scale, for numbers or tensors."""
    return before * (1.0 - momentum) + batch * (momentum * scale)


def select_matching_candidate(differences: dict[str, float], tolerance: float) -> str:
    """Returns the one candidate whose difference from the reference is within the tolerance.

    Raises:
        ReferenceMismatchError: No candidate is within the tolerance, or more than one is.
    """
    matching = [name for name, difference in differences.items() if difference <= tolerance]
    if len(matching) != 1:
        raise ReferenceMismatchError(
            f"{len(matching)} running-variance candidates lie within {tolerance:.3e} "
            f"of the reference: {differences}",
        )
    return matching[0]


def watch_batchnorm(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Returns a store the module's next forward call fills with the K_BATCHNORM_OBSERVED keys."""
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def record_before(owner: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        store["input"] = inputs[0].detach().clone()
        for leaf in ("weight", "bias"):
            store[leaf] = getattr(owner, leaf).detach().clone()
        for leaf in ("running_mean", "running_var"):
            store[f"before.{leaf}"] = getattr(owner, leaf).detach().clone()

    def record_after(owner: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        # The activation after the layer may run in place, so the output is copied here.
        store["output"] = output.detach().clone()
        for leaf in ("running_mean", "running_var"):
            store[f"after.{leaf}"] = getattr(owner, leaf).detach().clone()
        for handle in handles:
            handle.remove()

    handles.append(module.register_forward_pre_hook(record_before))
    handles.append(module.register_forward_hook(record_after))
    return store


def run_unit_batchnorm(seed: int) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Returns one BatchNorm2d with the Ultralytics settings and what one train-mode pass did."""
    import torch
    from ultralytics.utils.torch_utils import initialize_weights

    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.BatchNorm2d(K_BATCHNORM_UNIT_SHAPE[1])
    initialize_weights(layer)
    perturb_normalization(layer, generator)
    layer.train()
    observed = watch_batchnorm(layer)
    with torch.no_grad():
        layer(torch.randn(K_BATCHNORM_UNIT_SHAPE, generator=generator, dtype=torch.float32))
    return layer, observed


def export_batchnorm_case(
    case: str,
    module: torch.nn.Module,
    observed: dict[str, torch.Tensor],
) -> tuple[list[ExportedTensor], dict[str, Any]]:
    """Returns one observed pass and both candidates of its updated running variance.

    Raises:
        ReferenceMismatchError: Not exactly one candidate reproduces the reference's variance.
    """
    import torch

    prefix = f"{K_BATCHNORM_PREFIX}.{case}"
    source = f"torch.nn.BatchNorm2d, momentum {module.momentum}, eps {module.eps}, train mode"
    tensors = [
        export_tensor(f"{prefix}.{key}", observed[key], source, layout)
        for key, layout in K_BATCHNORM_OBSERVED.items()
    ]
    count = count_per_channel(observed["input"].shape)
    after = observed["after.running_var"]
    before = observed["before.running_var"].double()
    variance = observed["input"].double().var(dim=(0, 2, 3), unbiased=False)
    candidate_source = f"{Path(__file__).name} (export_batchnorm_case)"
    candidates = {}
    for candidate in K_BATCHNORM_CANDIDATES:
        scale = scale_batch_variance(candidate, count)
        value = blend_running_statistic(before, variance, module.momentum, scale).float()
        name = f"{prefix}.candidate.{candidate}.running_var"
        tensors.append(export_tensor(name, value, candidate_source, K_LAYOUT_VECTOR))
        candidates[candidate] = {
            "tensor": name,
            "batch_variance_scale": scale,
            "max_abs_difference": float((value.double() - after.double()).abs().max()),
            "equal_in_float32": bool(torch.equal(value, after)),
        }
    spacing = float(torch.finfo(torch.float32).eps) * float(after.abs().max())
    tolerance = K_BATCHNORM_MATCH_ULPS * spacing
    differences = {name: entry["max_abs_difference"] for name, entry in candidates.items()}
    context = {
        "module": f"torch.nn.{type(module).__name__}",
        "input_shape_nchw": [int(extent) for extent in observed["input"].shape],
        "values_per_channel": count,
        "momentum": module.momentum,
        "eps": module.eps,
        "candidates": candidates,
        "match_tolerance": tolerance,
        "matching_candidate": select_matching_candidate(differences, tolerance),
    }
    return tensors, context


def export_batchnorm(
    seed: int,
    model: torch.nn.Module,
    model_observed: dict[str, torch.Tensor],
) -> ReferenceFile:
    """Returns the batchnorm file: the unit case and one layer of the model's train-mode pass.

    Args:
        model_observed: Store of watch_batchnorm on K_BATCHNORM_MODEL_MODULE, filled by that pass.
    """
    from ultralytics.utils.torch_utils import initialize_weights

    passes = {
        K_BATCHNORM_UNIT_CASE: run_unit_batchnorm(seed),
        K_BATCHNORM_MODEL_MODULE: (model.get_submodule(K_BATCHNORM_MODEL_MODULE), model_observed),
    }
    tensors: list[ExportedTensor] = []
    cases = {}
    for case, (module, observed) in passes.items():
        written, cases[case] = export_batchnorm_case(case, module, observed)
        tensors.extend(written)
    context = {
        "mode": "train",
        "settings_source": describe_source(initialize_weights),
        "candidate_rule": "before.running_var * (1 - momentum) "
        "+ biased batch variance * momentum * batch_variance_scale, in float64, then float32",
        "max_abs_difference": "largest |candidate - after.running_var| over the channels, "
        "both read as float32",
        "match_tolerance": f"{K_BATCHNORM_MATCH_ULPS} * float32 epsilon * max |after.running_var|",
        "unit_case_state": "weight, bias and running statistics drawn as batch_norm_perturbation "
        f"of {K_WEIGHTS_FILE}; input drawn standard normal",
        "model_case_pass": "the train-mode pass of loss.sacp",
        "cases": cases,
    }
    return ReferenceFile(K_BATCHNORM_FILE, tuple(tensors), context)


def collect_run_facts(seed: int, commit: str) -> dict[str, Any]:
    """Returns the versions and settings the run used, read from the loaded libraries."""
    import torch
    import ultralytics

    return {
        "generator": f"tools/reference/{Path(__file__).name}",
        "ultralytics_commit": commit,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": seed,
        "thread_count": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "model_yaml": f"ultralytics/cfg/models/26/yolo26.yaml, resolved from {K_MODEL_YAML}",
        "model_scale": K_MODEL_SCALE,
        "class_count": K_CLASS_COUNT,
    }


def produce_references(seed: int, bn_state: str) -> tuple[dict[str, Any], list[ReferenceFile]]:
    """Returns the run facts and the five reference files of one seeded run."""
    import torch

    commit = read_installed_commit()
    model = build_model(seed)
    generator = torch.Generator().manual_seed(seed)
    if bn_state == "perturbed":
        perturb_normalization(model, generator)
    image_nhwc = make_input(generator)
    image_nchw = image_nhwc.permute(0, 3, 1, 2).contiguous()
    files = [
        export_weights(model, bn_state),
        export_input(image_nhwc),
        export_forward(model, image_nchw),
    ]
    # The watch is set after the eval-mode pass, so the train-mode pass is the one it records.
    observed = watch_batchnorm(model.get_submodule(K_BATCHNORM_MODEL_MODULE))
    files.append(export_loss(model, image_nchw, make_training_batch()))
    files.append(export_batchnorm(seed, model, observed))
    return collect_run_facts(seed, commit), files


def run(tests_dir: Path, seed: int, bn_state: str, external_weights: Path | None) -> None:
    """Produces the references, writes them and prints one line per written file.

    Raises:
        ExportError: The export failed; the message names the cause.
    """
    attempts = guard_network()
    configure_environment()
    try:
        load_torch(seed)
        facts, files = produce_references(seed, bn_state)
    except K_FOREIGN_ERRORS as error:
        detail = str(error).strip().splitlines()[0] if str(error).strip() else ""
        raise ExportError(f"{type(error).__name__}: {detail}") from error
    if attempts:
        raise NetworkAccessError(f"the reference code tried the network: {'; '.join(attempts)}")
    payloads = encode_files(files)
    manifest = build_manifest(facts, files, payloads, external_weights is not None)
    try:
        sizes = write_reference_files(payloads, manifest, tests_dir, external_weights)
    except OSError as error:
        raise ExportError(f"cannot write the references: {error}") from error
    for reference in files:
        print(f"{reference.name}: {len(reference.tensors)} tensors, {sizes[reference.name]} bytes")
    print(f"{K_MANIFEST_NAME}: {sizes[K_MANIFEST_NAME]} bytes")
    print(f"total: {sum(sizes.values())} bytes")


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description="Exports YOLO26n reference tensors as SACP files.")
    parser.add_argument("tests_dir", type=Path, help="tests/ folder holding the fixture folders")
    parser.add_argument("--seed", type=int, default=K_DEFAULT_SEED, help="seed of weights, input")
    parser.add_argument(
        "--bn-state",
        choices=("perturbed", "initial"),
        default="perturbed",
        help="perturbed draws every BatchNorm tensor from the seed; initial keeps torch's defaults",
    )
    parser.add_argument(
        "--external-weights",
        type=Path,
        default=None,
        help=f"directory for {K_WEIGHTS_FILE} when it must stay outside the fixture folders",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the export and returns the process exit code."""
    arguments = parse_arguments(argv)
    try:
        run(arguments.tests_dir, arguments.seed, arguments.bn_state, arguments.external_weights)
    except ExportError as error:
        print(f"export_yolo26_reference: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
