# make_caffe_lenet_fixture.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Writes the Caffe fixture of the Caffe importer: a LeNet prototxt and a seeded caffemodel.

The layers are those of Caffe's examples/mnist LeNet at widths of 4, 8 and 32 channels. The
caffemodel is encoded by the protobuf runtime over a descriptor of the caffe.proto field numbers.
Usage: python tools/reference/make_caffe_lenet_fixture.py OUTPUT_DIR [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

K_PROTOTXT_FILE = "lenet.prototxt"
K_MODEL_FILE = "lenet.caffemodel"
K_NET_NAME = "LeNet"
K_INPUT_NAME = "data"
K_INPUT_SHAPE = (1, 1, 28, 28)
K_KERNEL = 5
K_POOL = 2
K_FIRST_FILTERS = 4
K_SECOND_FILTERS = 8
K_HIDDEN_UNITS = 32
K_CLASSES = 10
K_DEFAULT_SEED = 7
K_WEIGHT_SCALE = 0.2

K_NET_NAME_FIELD = 1
K_NET_LAYER_FIELD = 100
K_LAYER_NAME_FIELD = 1
K_LAYER_TYPE_FIELD = 2
K_LAYER_BOTTOM_FIELD = 3
K_LAYER_TOP_FIELD = 4
K_LAYER_BLOBS_FIELD = 7
K_BLOB_DATA_FIELD = 5
K_BLOB_SHAPE_FIELD = 7
K_SHAPE_DIM_FIELD = 1

K_PROTOTXT = """name: "{name}"
input: "{input}"
input_shape {{
  dim: {n}
  dim: {c}
  dim: {h}
  dim: {w}
}}
layer {{
  name: "conv1"
  type: "Convolution"
  bottom: "{input}"
  top: "conv1"
  convolution_param {{
    num_output: {first}
    kernel_size: {kernel}
    stride: 1
  }}
}}
layer {{
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {{
    pool: MAX
    kernel_size: {pool}
    stride: {pool}
  }}
}}
layer {{
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  convolution_param {{
    num_output: {second}
    kernel_size: {kernel}
    stride: 1
  }}
}}
layer {{
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {{
    pool: MAX
    kernel_size: {pool}
    stride: {pool}
  }}
}}
layer {{
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  inner_product_param {{
    num_output: {hidden}
  }}
}}
layer {{
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}}
layer {{
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  inner_product_param {{
    num_output: {classes}
  }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
}}
"""


class FixtureError(Exception):
    """Reports a failure to build or write the Caffe fixture."""


def _import_numpy() -> Any:
    """Returns numpy, or raises FixtureError."""
    try:
        import numpy
    except ImportError as error:
        raise FixtureError(f"the fixture needs numpy: {error}") from error
    return numpy


def _import_protobuf() -> tuple[Any, Any, Any]:
    """Returns descriptor_pb2, descriptor_pool and message_factory, or raises FixtureError."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as error:
        raise FixtureError(f"the fixture needs the protobuf package: {error}") from error
    return descriptor_pb2, descriptor_pool, message_factory


def pooled_extent(extent: int) -> int:
    """Returns the extent a valid convolution and a pool give: (extent - kernel + 1) / pool."""
    return (extent - K_KERNEL + 1) // K_POOL


def inner_product_inputs() -> int:
    """Returns the features the first inner product reads: channels times the pooled extents."""
    height = pooled_extent(pooled_extent(K_INPUT_SHAPE[2]))
    width = pooled_extent(pooled_extent(K_INPUT_SHAPE[3]))
    return K_SECOND_FILTERS * height * width


def build_prototxt() -> str:
    """Returns the text of the deploy prototxt."""
    return K_PROTOTXT.format(
        name=K_NET_NAME,
        input=K_INPUT_NAME,
        n=K_INPUT_SHAPE[0],
        c=K_INPUT_SHAPE[1],
        h=K_INPUT_SHAPE[2],
        w=K_INPUT_SHAPE[3],
        first=K_FIRST_FILTERS,
        second=K_SECOND_FILTERS,
        hidden=K_HIDDEN_UNITS,
        classes=K_CLASSES,
        kernel=K_KERNEL,
        pool=K_POOL,
    )


def draw_weights(seed: int) -> list[tuple[str, str, list[Any]]]:
    """Returns (layer name, layer type, blobs) of each layer that holds weights, in net order.

    The blobs are float32 arrays drawn from a generator seeded with `seed`: the weight and the
    bias of each layer.
    """
    numpy = _import_numpy()
    generator = numpy.random.default_rng(seed)

    def normal(shape: tuple[int, ...]) -> Any:
        return generator.normal(0.0, K_WEIGHT_SCALE, size=shape).astype(numpy.float32)

    return [
        ("conv1", "Convolution",
         [normal((K_FIRST_FILTERS, K_INPUT_SHAPE[1], K_KERNEL, K_KERNEL)),
          normal((K_FIRST_FILTERS,))]),
        ("conv2", "Convolution",
         [normal((K_SECOND_FILTERS, K_FIRST_FILTERS, K_KERNEL, K_KERNEL)),
          normal((K_SECOND_FILTERS,))]),
        ("ip1", "InnerProduct",
         [normal((K_HIDDEN_UNITS, inner_product_inputs())), normal((K_HIDDEN_UNITS,))]),
        ("ip2", "InnerProduct", [normal((K_CLASSES, K_HIDDEN_UNITS)), normal((K_CLASSES,))]),
    ]


def message_classes() -> dict[str, Any]:
    """Returns the NetParameter, LayerParameter, BlobProto and BlobShape message classes.

    The classes come from a descriptor that states the fields of BVLC caffe.proto the fixture
    writes, under the numbers that file gives them.
    """
    descriptor_pb2, descriptor_pool, message_factory = _import_protobuf()
    field = descriptor_pb2.FieldDescriptorProto
    proto = descriptor_pb2.FileDescriptorProto(name="caffe_subset.proto", package="caffe")

    def add_field(message: Any, name: str, number: int, kind: int, repeated: bool,
                  type_name: str = "", packed: bool = False) -> None:
        added = message.field.add(
            name=name, number=number, type=kind,
            label=field.LABEL_REPEATED if repeated else field.LABEL_OPTIONAL,
        )
        if type_name:
            added.type_name = type_name
        if packed:
            added.options.packed = True

    shape = proto.message_type.add(name="BlobShape")
    add_field(shape, "dim", K_SHAPE_DIM_FIELD, field.TYPE_INT64, True, packed=True)
    blob = proto.message_type.add(name="BlobProto")
    add_field(blob, "data", K_BLOB_DATA_FIELD, field.TYPE_FLOAT, True, packed=True)
    add_field(blob, "shape", K_BLOB_SHAPE_FIELD, field.TYPE_MESSAGE, False, ".caffe.BlobShape")
    layer = proto.message_type.add(name="LayerParameter")
    add_field(layer, "name", K_LAYER_NAME_FIELD, field.TYPE_STRING, False)
    add_field(layer, "type", K_LAYER_TYPE_FIELD, field.TYPE_STRING, False)
    add_field(layer, "bottom", K_LAYER_BOTTOM_FIELD, field.TYPE_STRING, True)
    add_field(layer, "top", K_LAYER_TOP_FIELD, field.TYPE_STRING, True)
    add_field(layer, "blobs", K_LAYER_BLOBS_FIELD, field.TYPE_MESSAGE, True, ".caffe.BlobProto")
    net = proto.message_type.add(name="NetParameter")
    add_field(net, "name", K_NET_NAME_FIELD, field.TYPE_STRING, False)
    add_field(net, "layer", K_NET_LAYER_FIELD, field.TYPE_MESSAGE, True, ".caffe.LayerParameter")

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(proto.SerializeToString())
    return {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"caffe.{name}"))
        for name in ("NetParameter", "LayerParameter", "BlobProto", "BlobShape")
    }


def build_caffemodel(weights: list[tuple[str, str, list[Any]]]) -> bytes:
    """Returns the bytes of a caffemodel that holds `weights`, one layer message per entry."""
    classes = message_classes()
    net = classes["NetParameter"](name=K_NET_NAME)
    for layer_name, layer_type, blobs in weights:
        layer = net.layer.add(name=layer_name, type=layer_type)
        for array in blobs:
            blob = layer.blobs.add()
            blob.shape.dim.extend(int(extent) for extent in array.shape)
            blob.data.extend(float(value) for value in array.reshape(-1))
    return net.SerializeToString()


def write_fixture(directory: Path, seed: int = K_DEFAULT_SEED) -> tuple[Path, Path]:
    """Writes the prototxt and the caffemodel into `directory` and returns both paths."""
    prototxt = build_prototxt()
    model = build_caffemodel(draw_weights(seed))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        prototxt_path = directory / K_PROTOTXT_FILE
        model_path = directory / K_MODEL_FILE
        prototxt_path.write_text(prototxt, encoding="utf-8", newline="\n")
        model_path.write_bytes(model)
    except OSError as error:
        raise FixtureError(f"cannot write into {directory}: {error}") from error
    return prototxt_path, model_path


def main(argv: list[str]) -> int:
    """Parses the command line, writes the fixture and returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="receives the prototxt and the caffemodel")
    parser.add_argument("--seed", type=int, default=K_DEFAULT_SEED, help="seed of the weights")
    arguments = parser.parse_args(argv)
    try:
        prototxt_path, model_path = write_fixture(arguments.output_dir, arguments.seed)
    except FixtureError as error:
        print(f"make_caffe_lenet_fixture: {error}", file=sys.stderr)
        return 1
    print(f"wrote {prototxt_path} and {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
