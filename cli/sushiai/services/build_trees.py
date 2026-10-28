# build_trees.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the build tree of a build type and a sanitizer set.

The reasoning is in cli/README.md under "services/build_trees.py".
"""

from __future__ import annotations

from pathlib import Path

from sushicore.build_tree import tree_dir

from .sanitizer import Sanitizer

#: The tree of a build without sanitizers.
K_DEFAULT_TREE = "default"

#: The tree each sanitizer set builds in; the set is fixed by the tree.
K_SANITIZER_TREES = {
    Sanitizer.address_undefined: "asan",
    Sanitizer.thread: "tsan",
}


def tree_name(sanitizer: Sanitizer | None) -> str:
    """Returns the name of the tree *sanitizer* builds in; none or None is the default tree."""
    return K_SANITIZER_TREES.get(sanitizer, K_DEFAULT_TREE) if sanitizer else K_DEFAULT_TREE


def build_dir(root: Path, build_type: str, sanitizer: Sanitizer | None = None) -> Path:
    """Returns the build tree of *build_type* and *sanitizer* under *root*."""
    return tree_dir(root, tree_name(sanitizer), build_type)
