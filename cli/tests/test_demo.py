# test_demo.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests end to end that `sa demo mlp` runs, forwards its flags and compiles once.

The reasoning is in cli/tests/README.md under "test_demo.py".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_runs_and_reports_one_compile():
    """`sa demo mlp` exits 0 and reports exactly one compile."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout


# fp32 must report no casts and mixed must report some, through the same forwarding.
_PRECISION_CASES = [
    ("fp32", re.compile(r"Precision:\s*fp32;\s*every operation")),
    ("mixed", re.compile(r"Precision:\s*mixed;\s*[1-9]\d*\s*cast")),
]


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
@pytest.mark.parametrize("mode,expected", _PRECISION_CASES, ids=[c[0] for c in _PRECISION_CASES])
def test_sa_demo_mlp_precision_flag_reaches_the_demo(mode, expected):
    """`--precision fp32|mixed` is forwarded and changes what the banner says."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--precision", mode],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert expected.search(completed.stdout), completed.stdout

    # The compile-once contract holds in both precision modes.
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_loss_scale_flag_reaches_the_demo_and_is_exact():
    """`--loss-scale` is forwarded, reported, and changes no fp32 number."""
    loss = re.compile(r"^loss\s+:\s+(\S+) -> (\S+)$", re.MULTILINE)

    plain = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    scaled = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--loss-scale", "65536"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert scaled.returncode == 0, scaled.stdout + scaled.stderr

    assert "Scaling  : gradients scaled by 65536" in scaled.stdout, scaled.stdout
    assert "loss scale     : 65536" in scaled.stdout, scaled.stdout
    assert _COMPILE_ONCE.search(scaled.stdout), scaled.stdout

    plain_loss = loss.search(plain.stdout)
    scaled_loss = loss.search(scaled.stdout)
    assert plain_loss and scaled_loss, plain.stdout + scaled.stdout
    assert plain_loss.groups() == scaled_loss.groups(), (
        "scaling by a power of two moved the loss, so the unscale is not exact"
    )


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_rejects_a_non_dyadic_loss_scale():
    """A scale that is not a power of two fails rather than rounding quietly."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--loss-scale", "3"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode != 0
    assert "power of two" in (completed.stdout + completed.stderr)


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_rejects_an_unknown_precision():
    """An unrecognised `--precision` value fails rather than falling back."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--precision", "bfloat16"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode != 0
    assert "bfloat16" in (completed.stdout + completed.stderr)


# Each activation must fuse as its own form, which shows the right model was built.
_ACTIVATION_CASES = [
    ("relu", "gemm+bias+relu"),
    ("gelu", "gemm+bias+gelu"),
]


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
@pytest.mark.parametrize("activation,form", _ACTIVATION_CASES,
                         ids=[c[0] for c in _ACTIVATION_CASES])
def test_sa_demo_mlp_activation_flag_reaches_the_demo(activation, form):
    """`--activation relu|gelu` is forwarded and changes which form is fused."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--activation", activation],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"fused    {form}" in completed.stdout, completed.stdout
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_rejects_an_unknown_activation():
    """An unrecognised `--activation` value fails rather than falling back."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS, "--activation", "swish"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode != 0
    assert "swish" in (completed.stdout + completed.stderr)
