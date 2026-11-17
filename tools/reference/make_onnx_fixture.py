# make_onnx_fixture.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Writes the small ONNX fixture of the ONNX importer and the output the reference run gives.

The graph is a strided convolution, batch normalisation, a SiLU, a nearest Resize, a Concat, a
1x1 convolution and a Reshape, at opset 17. onnx.reference.ReferenceEvaluator computes the output.
Usage: see tools/reference/README.md, "make_onnx_fixture.py".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

K_MODEL_FILE = "silu_resize.onnx"
K_EXPECTED_FILE = "silu_resize.expected.json"
K_FORMAT_VERSION = 1
K_OPSET = 17
K_IR_VERSION = 8
K_INPUT_NAME = "images"
K_OUTPUT_NAME = "output"
K_INPUT_SHAPE = (1, 3, 8, 8)
K_STEM_CHANNELS = 4
K_JOIN_CHANNELS = 2
K_BN_EPSILON = 1e-3
K_DEFAULT_SEED = 7


class FixtureError(Exception):
    """Reports a failure to build or write the ONNX fixture."""


def _import_onnx() -> tuple[Any, Any, Any, Any]:
    """Returns numpy, onnx, onnx.helper and onnx.numpy_helper, or raises FixtureError."""
    try:
        import numpy
        import onnx
        from onnx import helper, numpy_helper
        from onnx.reference import ReferenceEvaluator  # noqa: F401
    except ImportError as error:
        message = f"the fixture needs numpy and onnx with onnx.reference: {error}"
        raise FixtureError(message) from error
    return numpy, onnx, helper, numpy_helper


def draw_weights(seed: int) -> dict[str, Any]:
    """Returns the float32 weights of the graph, drawn from a generator seeded with `seed`."""
    numpy, _, _, _ = _import_onnx()
    generator = numpy.random.default_rng(seed)

    def normal(shape: tuple[int, ...]) -> Any:
        return generator.normal(0.0, 0.5, size=shape).astype(numpy.float32)

    return {
        "stem.weight": normal((K_STEM_CHANNELS, K_INPUT_SHAPE[1], 3, 3)),
        "stem.bias": normal((K_STEM_CHANNELS,)),
        "bn.scale": generator.uniform(0.5, 1.5, K_STEM_CHANNELS).astype(numpy.float32),
        "bn.bias": normal((K_STEM_CHANNELS,)),
        "bn.mean": normal((K_STEM_CHANNELS,)),
        "bn.var": generator.uniform(0.5, 1.5, K_STEM_CHANNELS).astype(numpy.float32),
        "join.weight": normal((K_JOIN_CHANNELS, K_STEM_CHANNELS + K_INPUT_SHAPE[1], 1, 1)),
    }


def fixed_input() -> Any:
    """Returns the one input the expected output is computed on: 0, 1, 2, ... scaled to [0, 1)."""
    numpy, _, _, _ = _import_onnx()
    count = 1
    for extent in K_INPUT_SHAPE:
        count *= extent
    return (numpy.arange(count, dtype=numpy.float32) / numpy.float32(count)).reshape(K_INPUT_SHAPE)


def _array(dtype: str, values: list[float] | list[int]) -> Any:
    """Returns a one-dimensional numpy array of `values` in `dtype`."""
    numpy, _, _, _ = _import_onnx()
    return numpy.asarray(values, dtype=dtype)


def build_model(weights: dict[str, Any]) -> Any:
    """Returns the ModelProto of the fixture over `weights`."""
    _, onnx, helper, numpy_helper = _import_onnx()
    nodes = [
        helper.make_node(
            "Conv", [K_INPUT_NAME, "stem.weight", "stem.bias"], ["stem"], name="stem",
            kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "BatchNormalization", ["stem", "bn.scale", "bn.bias", "bn.mean", "bn.var"], ["bn"],
            name="bn", epsilon=K_BN_EPSILON,
        ),
        helper.make_node("Sigmoid", ["bn"], ["gate"], name="gate"),
        helper.make_node("Mul", ["bn", "gate"], ["silu"], name="silu"),
        helper.make_node(
            "Resize", ["silu", "", "up.scales"], ["up"], name="up", mode="nearest",
            coordinate_transformation_mode="asymmetric", nearest_mode="floor",
        ),
        helper.make_node("Concat", ["up", K_INPUT_NAME], ["joined"], name="joined", axis=1),
        helper.make_node(
            "Conv", ["joined", "join.weight"], ["join"], name="join", kernel_shape=[1, 1],
        ),
        helper.make_node("Reshape", ["join", "flat.shape"], [K_OUTPUT_NAME], name="flat"),
    ]
    initializers = [numpy_helper.from_array(array, name) for name, array in weights.items()]
    initializers.append(
        numpy_helper.from_array(_array("float32", [1.0, 1.0, 2.0, 2.0]), "up.scales")
    )
    initializers.append(numpy_helper.from_array(_array("int64", [1, -1]), "flat.shape"))
    flat = K_JOIN_CHANNELS * K_INPUT_SHAPE[2] * K_INPUT_SHAPE[3]
    graph = helper.make_graph(
        nodes,
        "silu_resize",
        [helper.make_tensor_value_info(K_INPUT_NAME, onnx.TensorProto.FLOAT, list(K_INPUT_SHAPE))],
        [helper.make_tensor_value_info(K_OUTPUT_NAME, onnx.TensorProto.FLOAT, [1, flat])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", K_OPSET)], producer_name="make_onnx_fixture"
    )
    model.ir_version = K_IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def evaluate(model: Any, image: Any) -> Any:
    """Returns the output onnx.reference.ReferenceEvaluator computes for `image`."""
    _import_onnx()
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    (output,) = evaluator.run(None, {K_INPUT_NAME: image})
    return output


def expected_record(image: Any, output: Any) -> dict[str, Any]:
    """Returns the JSON document of the input and the output, values in row-major NCHW order."""
    return {
        "format_version": K_FORMAT_VERSION,
        "opset": K_OPSET,
        "input": {
            "name": K_INPUT_NAME,
            "shape": list(image.shape),
            "values": [float(value) for value in image.reshape(-1)],
        },
        "output": {
            "name": K_OUTPUT_NAME,
            "shape": list(output.shape),
            "values": [float(value) for value in output.reshape(-1)],
        },
    }


def write_fixture(directory: Path, seed: int = K_DEFAULT_SEED) -> tuple[Path, Path]:
    """Writes the model and its expected output into `directory` and returns both paths."""
    _, onnx, _, _ = _import_onnx()
    model = build_model(draw_weights(seed))
    image = fixed_input()
    output = evaluate(model, image)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / K_MODEL_FILE
        expected_path = directory / K_EXPECTED_FILE
        onnx.save(model, str(model_path))
        expected_path.write_text(
            json.dumps(expected_record(image, output), indent=1) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise FixtureError(f"cannot write into {directory}: {error}") from error
    return model_path, expected_path


def main(argv: list[str]) -> int:
    """Parses the command line, writes the fixture and returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="receives the .onnx and the expected JSON")
    parser.add_argument("--seed", type=int, default=K_DEFAULT_SEED, help="seed of the weights")
    arguments = parser.parse_args(argv)
    try:
        model_path, expected_path = write_fixture(arguments.output_dir, arguments.seed)
    except FixtureError as error:
        print(f"make_onnx_fixture: {error}", file=sys.stderr)
        return 1
    print(f"wrote {model_path} and {expected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
