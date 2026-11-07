# test_demo.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests end to end that `sa demo mlp` runs, forwards its flags and compiles once.

The reasoning is in cli/tests/README.md under "test_demo.py".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from checkout_cli import K_DEFAULT_TREE, run_cli

pytestmark = pytest.mark.medium

# cli/tests/test_demo.py -> cli/tests -> cli -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# The demo prints the count twice; either match is enough, and the number must be 1.
_COMPILE_ONCE = re.compile(r"compiles\s*:\s*1\b")

# --mnist-dir names a directory that cannot exist, so the synthetic fallback always runs.
_DEMO_ARGS = [
    "--epochs", "1",
    "--batches", "2",
    "--batch-size", "8",
    "--hidden", "16",
    "--features", "6",
    "--classes", "4",
    "--mnist-dir", "does-not-exist",
]


_needs_build = pytest.mark.skipif(
    not K_DEFAULT_TREE.is_dir(), reason="no build tree — run `sa build` first")


def _demo(*extra: str) -> subprocess.CompletedProcess:
    """Runs `sa demo mlp` with the tiny arguments and *extra* from the repository root."""
    return run_cli("demo", "mlp", *_DEMO_ARGS, *extra, cwd=REPO_ROOT, timeout=900)


def _output(completed: subprocess.CompletedProcess) -> str:
    """Returns what the run printed on both streams."""
    return completed.stdout + completed.stderr


@_needs_build
@pytest.mark.parametrize("flags", [
    [],
    ["--precision", "fp32"],
    ["--precision", "mixed"],
    ["--loss-scale", "65536"],
    ["--activation", "relu"],
    ["--activation", "gelu"],
], ids=["default", "fp32", "mixed", "loss-scale", "relu", "gelu"])
def test_sa_demo_mlp_runs_and_reports_one_compile(flags):
    """`sa demo mlp` exits 0 and reports exactly one compile, in every mode it offers."""
    completed = _demo(*flags)

    assert completed.returncode == 0, _output(completed)
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout


# fp32 must report no casts and mixed must report some, through the same forwarding.
_PRECISION_CASES = [
    ("fp32", re.compile(r"Precision:\s*fp32;\s*every operation")),
    ("mixed", re.compile(r"Precision:\s*mixed;\s*[1-9]\d*\s*cast")),
]


@_needs_build
@pytest.mark.parametrize("mode,expected", _PRECISION_CASES, ids=[c[0] for c in _PRECISION_CASES])
def test_sa_demo_mlp_precision_flag_reaches_the_demo(mode, expected):
    """`--precision fp32|mixed` is forwarded and changes what the banner says."""
    completed = _demo("--precision", mode)

    assert completed.returncode == 0, _output(completed)
    assert expected.search(completed.stdout), completed.stdout


@_needs_build
def test_sa_demo_mlp_loss_scale_flag_is_forwarded_and_reported():
    """`--loss-scale 65536` reaches the demo, which reports the scale twice."""
    scaled = _demo("--loss-scale", "65536")

    assert scaled.returncode == 0, _output(scaled)
    assert "Scaling  : gradients scaled by 65536" in scaled.stdout, scaled.stdout
    assert "loss scale     : 65536" in scaled.stdout, scaled.stdout


@_needs_build
def test_sa_demo_mlp_loss_scale_by_a_power_of_two_changes_no_fp32_number():
    """A dyadic scale is exact, so the scaled loss equals the unscaled one digit for digit."""
    loss = re.compile(r"^loss\s+:\s+(\S+) -> (\S+)$", re.MULTILINE)
    plain = _demo()
    scaled = _demo("--loss-scale", "65536")

    assert plain.returncode == 0, _output(plain)
    assert scaled.returncode == 0, _output(scaled)
    plain_loss = loss.search(plain.stdout)
    scaled_loss = loss.search(scaled.stdout)
    assert plain_loss and scaled_loss, plain.stdout + scaled.stdout
    assert plain_loss.groups() == scaled_loss.groups(), (
        "scaling by a power of two moved the loss, so the unscale is not exact"
    )


@_needs_build
def test_sa_demo_mlp_rejects_a_non_dyadic_loss_scale():
    """A scale that is not a power of two fails rather than rounding quietly."""
    completed = _demo("--loss-scale", "3")

    assert completed.returncode != 0
    assert "power of two" in _output(completed)


@_needs_build
def test_sa_demo_mlp_rejects_an_unknown_precision():
    """An unrecognised `--precision` value fails rather than falling back."""
    completed = _demo("--precision", "bfloat16")

    assert completed.returncode != 0
    assert "bfloat16" in _output(completed)


# Each activation must fuse as its own form, which shows the right model was built.
_ACTIVATION_CASES = [
    ("relu", "gemm+bias+relu"),
    ("gelu", "gemm+bias+gelu"),
]


@_needs_build
@pytest.mark.parametrize("activation,form", _ACTIVATION_CASES,
                         ids=[c[0] for c in _ACTIVATION_CASES])
def test_sa_demo_mlp_activation_flag_reaches_the_demo(activation, form):
    """`--activation relu|gelu` is forwarded and changes which form is fused."""
    completed = _demo("--activation", activation)

    assert completed.returncode == 0, _output(completed)
    assert f"fused    {form}" in completed.stdout, completed.stdout


@_needs_build
def test_sa_demo_mlp_rejects_an_unknown_activation():
    """An unrecognised `--activation` value fails rather than falling back."""
    completed = _demo("--activation", "swish")

    assert completed.returncode != 0
    assert "swish" in _output(completed)
