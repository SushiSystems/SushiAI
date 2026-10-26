# sycl_targets.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the variable `sa build --sycl-targets` gives the SushiBLAS package and keeps the list in the tree."""

from __future__ import annotations

from pathlib import Path

#: The SushiBLAS cache variable that lists the ahead-of-time device targets.
K_SYCL_TARGETS_VARIABLE = "SB_SYCL_TARGETS"

#: The file of a build tree that keeps the list its last build gave the SushiBLAS package.
K_TARGETS_FILE = ".sushiai_sycl_targets"


def sycl_targets_define(targets: str | None) -> list[str]:
    """Returns the VAR=VALUE entry that sets SB_SYCL_TARGETS.

    Args:
        targets: The semicolon-separated target list, an empty string to clear it, None to
            leave the cached value.
    """
    if targets is None:
        return []
    return [f"{K_SYCL_TARGETS_VARIABLE}={targets}"]


def record_sycl_targets(tree: Path, targets: str) -> None:
    """Write *targets* into *tree*, for a later build that leaves the option out."""
    tree.mkdir(parents=True, exist_ok=True)
    (tree / K_TARGETS_FILE).write_text(targets, encoding="utf-8")


def read_sycl_targets(tree: Path) -> str:
    """Return the list :func:`record_sycl_targets` wrote for *tree*, none when it wrote none."""
    try:
        return (tree / K_TARGETS_FILE).read_text(encoding="utf-8")
    except OSError:
        return ""
