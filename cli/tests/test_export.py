# test_export.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests end to end that `sushiai export` writes files `sushiai import` reads back.

The reasoning is in cli/tests/README.md under "test_export.py".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from checkout_cli import K_DEFAULT_TREE, run_cli

from sushiai.services import discovery

pytestmark = pytest.mark.medium

# cli/tests/test_export.py -> cli/tests -> cli -> <repo root>
K_REPO_ROOT = Path(__file__).resolve().parents[2]
K_BUILD_DIR = K_DEFAULT_TREE
K_PROGRAM_TARGETS = ("sushiai_import", "sushiai_export")
K_ONNX_FIXTURE = (
    K_REPO_ROOT / "tests" / "vision" / "importers" / "fixtures" / "onnx" / "silu_resize.onnx")

# A 32 by 32 network: one normalised convolution, one plain convolution and one head.
K_SMALL_CFG = (
    "[net]\nwidth=32\nheight=32\nchannels=3\n"
    "[convolutional]\nbatch_normalize=1\nfilters=4\nsize=3\nstride=1\npad=1\n"
    "activation=leaky\n"
    "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=1\nactivation=linear\n"
    "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n"
)


def _are_programs_built() -> bool:
    """Returns whether the build tree holds the import and the export program."""
    if not K_BUILD_DIR.is_dir():
        return False
    return all(discovery.match_by_name(K_BUILD_DIR, target) is not None
               for target in K_PROGRAM_TARGETS)


_needs_build = pytest.mark.skipif(
    not _are_programs_built(),
    reason="sushiai_import or sushiai_export is not built; run `sa build` first")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Runs this checkout's CLI from the repository root, decoding its output as UTF-8."""
    return run_cli(*args, cwd=K_REPO_ROOT, timeout=900)


def _combined(completed: subprocess.CompletedProcess) -> str:
    """Returns what the command printed on both streams."""
    return completed.stdout + completed.stderr


@pytest.fixture
def darknet_model(tmp_path: Path) -> Path:
    """Imports the small configuration without weights and returns the folder it wrote."""
    cfg = tmp_path / "small.cfg"
    cfg.write_text(K_SMALL_CFG, encoding="utf-8", newline="\n")
    output = tmp_path / "darknet_model"
    completed = _run("import", "darknet", str(cfg), "--output", str(output))
    assert completed.returncode == 0, _combined(completed)
    return output


@pytest.fixture
def onnx_model(tmp_path: Path) -> Path:
    """Imports the ONNX fixture, weights included, and returns the folder it wrote."""
    output = tmp_path / "onnx_model"
    completed = _run("import", "onnx", str(K_ONNX_FIXTURE), "--output", str(output))
    assert completed.returncode == 0, _combined(completed)
    return output


@_needs_build
def test_export_writes_a_model_without_weights_that_imports_back(
    darknet_model: Path, tmp_path: Path,
):
    """The first usage line without a weight file: the .cfg is written and imports again."""
    exported = tmp_path / "exported"
    completed = _run("export", "darknet", str(darknet_model / "model.json"),
                     "--output", str(exported), "--training", str(darknet_model / "train.json"),
                     "--name", "small")

    assert completed.returncode == 0, _combined(completed)
    assert (exported / "small.cfg").is_file()
    back = _run("import", "darknet", str(exported / "small.cfg"),
                "--output", str(tmp_path / "back"))
    assert back.returncode == 0, _combined(back)


@_needs_build
def test_export_refuses_a_published_flatten_and_names_the_row(onnx_model: Path, tmp_path: Path):
    """The ONNX fixture publishes a flatten; the .onnx is refused with its row named."""
    exported = tmp_path / "exported"
    completed = _run("export", "onnx", str(onnx_model / "model.json"),
                     str(onnx_model / "weights.sushi"), "--output", str(exported))

    assert completed.returncode != 0
    assert "row 6 'flatten'" in _combined(completed)
    assert not (exported / "model.onnx").exists()


@_needs_build
@pytest.mark.parametrize("weight_format, file_name",
                         [("safetensors", "model.safetensors"), ("torch", "model.pth")])
def test_export_weights_writes_a_file_that_imports_back(
    onnx_model: Path, tmp_path: Path, weight_format: str, file_name: str,
):
    """The second usage line: each weight format is written and `import weights` reads it."""
    written = tmp_path / file_name
    completed = _run("export", "weights", weight_format, str(onnx_model / "model.json"),
                     str(onnx_model / "weights.sushi"), "--output", str(written))

    assert completed.returncode == 0, _combined(completed)
    assert written.is_file()
    name_map = tmp_path / "map.json"
    name_map.write_text("{}", encoding="utf-8")
    back = _run("import", "weights", str(onnx_model / "model.json"), str(written),
                "--map", str(name_map), "--output", str(tmp_path / "back.sushi"))
    assert back.returncode == 0, _combined(back)


@_needs_build
def test_export_refuses_to_replace_a_file_without_force(darknet_model: Path, tmp_path: Path):
    """A second export into the same directory stops; --force lets it through."""
    arguments = ("export", "darknet", str(darknet_model / "model.json"),
                 "--output", str(tmp_path / "exported"))
    assert _run(*arguments).returncode == 0

    refused = _run(*arguments)
    assert refused.returncode != 0
    assert "model.cfg" in _combined(refused), _combined(refused)
    forced = _run(*arguments, "--force")
    assert forced.returncode == 0, _combined(forced)


@_needs_build
def test_an_unknown_format_lists_the_formats_the_build_has(darknet_model: Path, tmp_path: Path):
    """The C++ diagnostic reaches the caller and names the three exporters."""
    completed = _run("export", "tflite", str(darknet_model / "model.json"),
                     "--output", str(tmp_path / "none"))

    assert completed.returncode != 0
    combined = _combined(completed)
    assert "'tflite'" in combined, combined
    for name in ("onnx", "caffe", "darknet"):
        assert name in combined, combined
