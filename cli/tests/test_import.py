# test_import.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests end to end that `sushiai import` writes the files `sushiai train` reads.

The reasoning is in cli/tests/README.md under "test_import.py".
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from checkout_cli import K_DEFAULT_TREE, run_cli

from sushiai.services import discovery

pytestmark = pytest.mark.medium

# cli/tests/test_import.py -> cli/tests -> cli -> <repo root>
K_REPO_ROOT = Path(__file__).resolve().parents[2]
K_BUILD_DIR = K_DEFAULT_TREE
K_IMPORT_TARGET = "sushiai_import"

# A 32 by 32 network: one normalised convolution, one plain convolution and one head.
K_SMALL_CFG = (
    "[net]\nbatch=8\nsubdivisions=2\nwidth=32\nheight=32\nchannels=3\nmomentum=0.9\n"
    "decay=0.0005\nlearning_rate=0.001\nburn_in=100\nmax_batches=2000\npolicy=steps\n"
    "steps=1600,1800\nscales=.1,.1\n"
    "[convolutional]\nbatch_normalize=1\nfilters=4\nsize=3\nstride=1\npad=1\n"
    "activation=leaky\n"
    "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=1\nactivation=linear\n"
    "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\njitter=.3\n"
    "ignore_thresh=.7\ntruth_thresh=1\nrandom=1\n"
)


def _is_import_built() -> bool:
    """Returns whether the build tree holds the import program."""
    if not K_BUILD_DIR.is_dir():
        return False
    return discovery.match_by_name(K_BUILD_DIR, K_IMPORT_TARGET) is not None


_needs_build = pytest.mark.skipif(
    not _is_import_built(), reason="sushiai_import is not built; run `sa build` first")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Runs this checkout's CLI from the repository root, decoding its output as UTF-8."""
    return run_cli(*args, cwd=K_REPO_ROOT, timeout=900)


@pytest.fixture
def small_cfg(tmp_path: Path) -> Path:
    """Returns the path of the small configuration, written into the test's own folder."""
    path = tmp_path / "small.cfg"
    path.write_text(K_SMALL_CFG, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def imported(small_cfg: Path, tmp_path: Path) -> Path:
    """Imports the small configuration without weights and returns the folder it wrote."""
    output = tmp_path / "imported"
    completed = _run("import", "darknet", str(small_cfg), "--output", str(output))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


@_needs_build
def test_import_writes_a_model_file_that_carries_the_anchor_task(imported: Path):
    """A .cfg without weights becomes a model.json whose task is anchor detection."""
    model = json.loads((imported / "model.json").read_text(encoding="utf-8"))
    assert model["task"]["type"] == "anchor_detection"


@_needs_build
def test_import_writes_a_training_file_with_the_anchor_objective_and_no_data(imported: Path):
    """The train.json carries the anchor objective and `"data": null` for the user to fill."""
    training = json.loads((imported / "train.json").read_text(encoding="utf-8"))
    assert training["objective"]["type"] == "anchor"
    assert training["data"] is None


@_needs_build
def test_import_without_weights_writes_no_weight_file(imported: Path):
    """No weight file is written, and the training file names none."""
    training = json.loads((imported / "train.json").read_text(encoding="utf-8"))
    assert "weights" not in training
    assert not (imported / "weights.sushi").exists()


@_needs_build
def test_import_prints_a_note_for_each_key_it_cannot_place(small_cfg: Path, tmp_path: Path):
    """subdivisions and max_batches have no key in a training file, and the output says so."""
    completed = _run("import", "darknet", str(small_cfg), "--output", str(tmp_path / "noted"))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "subdivisions=2" in completed.stdout, completed.stdout
    assert "max_batches=2000" in completed.stdout, completed.stdout


@_needs_build
def test_train_refuses_the_imported_file_until_its_data_is_filled(
    small_cfg: Path, tmp_path: Path,
):
    """The training file states `"data": null`, and `train` names that key when it stops."""
    output = tmp_path / "unfilled"
    assert _run("import", "darknet", str(small_cfg), "--output", str(output)).returncode == 0

    completed = _run("train", str(output / "train.json"))

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "'data'" in combined, combined


@_needs_build
def test_import_refuses_to_replace_a_file_without_force(small_cfg: Path, imported: Path):
    """A second import into the same directory stops and names the file that exists."""
    refused = _run("import", "darknet", str(small_cfg), "--output", str(imported))

    assert refused.returncode != 0
    assert "exists already" in refused.stdout + refused.stderr


@_needs_build
def test_import_replaces_a_file_with_force(small_cfg: Path, imported: Path):
    """--force lets a second import into the same directory through."""
    forced = _run("import", "darknet", str(small_cfg), "--output", str(imported), "--force")

    assert forced.returncode == 0, forced.stdout + forced.stderr


@_needs_build
def test_an_unknown_format_lists_the_formats_the_build_has(small_cfg: Path, tmp_path: Path):
    """The C++ diagnostic reaches the caller and names `darknet`."""
    completed = _run("import", "caffe", str(small_cfg), "--output", str(tmp_path / "none"))

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "'caffe'" in combined and "'darknet'" in combined, combined
