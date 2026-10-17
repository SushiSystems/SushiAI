# test_train.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests end to end that `sushiai train` and `sushiai inference` run and compile once.

The reasoning is in cli/tests/README.md under "test_train.py".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from checkout_cli import run_cli

pytestmark = pytest.mark.medium

# cli/tests/test_train.py -> cli/tests -> cli -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

_COMPILE_ONCE = re.compile(r"compiles\s*:\s*1\b")

# tests/applications/programs/CMakeLists.txt writes tiny_train.json into the build tree.
_TINY_RUN = REPO_ROOT / "tests" / "applications" / "programs" / "fixtures" / "tiny_run"
_BUILT_TINY_RUN = REPO_ROOT / "build" / _TINY_RUN.relative_to(REPO_ROOT)
_TRAIN_CONFIG = _BUILT_TINY_RUN / "tiny_train.json"
_CHECKPOINT = _BUILT_TINY_RUN / "tiny.sushi"
_MODEL_CONFIG = _TINY_RUN / "tiny_model.json"
_EVAL_CONFIG = _TINY_RUN / "tiny_eval.json"

_needs_build = pytest.mark.skipif(
    not _TRAIN_CONFIG.is_file(), reason="no build tree — run `sa build` first")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Runs this checkout's CLI from the repository root, decoding its output as UTF-8."""
    return run_cli(*args, cwd=REPO_ROOT, timeout=900)


@_needs_build
def test_sushiai_train_runs_and_reports_one_compile():
    """`sushiai train <train.json>` exits 0 and compiles the step exactly once."""
    completed = _run("train", str(_TRAIN_CONFIG))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout
    assert _CHECKPOINT.is_file(), "train did not write the checkpoint it was asked for"


@_needs_build
def test_sushiai_inference_loads_the_checkpoint_train_wrote():
    """`sushiai inference` reads a checkpoint and reports one compile."""
    # Training runs here too, so the checkpoint does not depend on collection order.
    _run("train", str(_TRAIN_CONFIG))

    completed = _run("inference", str(_MODEL_CONFIG), str(_CHECKPOINT), str(_EVAL_CONFIG))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout


@_needs_build
def test_both_commands_appear_in_the_top_level_help():
    """They are subcommands of the same app `sa` is, not a separate program."""
    completed = _run("--help")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "train" in completed.stdout
    assert "inference" in completed.stdout


@_needs_build
def test_a_wrong_document_kind_is_refused_with_the_librarys_own_message():
    """The C++ diagnostic reaches the caller rather than being swallowed.

    A model.json handed to `train` must not be read as a train.json: every
    configuration document leads with its version key precisely so that this
    is an error rather than a misread.
    """
    completed = _run("train", str(_MODEL_CONFIG))

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "sushiai_train" in combined, combined


@_needs_build
def test_a_missing_config_names_the_path():
    """An absent file is refused by name rather than by a generic failure."""
    completed = _run("train", "no/such/train.json")

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "no/such/train.json" in combined, combined
