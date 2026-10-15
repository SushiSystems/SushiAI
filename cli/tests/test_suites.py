# test_suites.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the Suite enum and the two suites that are not CTest labels.

The reasoning is in cli/tests/README.md under "test_suites.py".
"""

from __future__ import annotations

from sushiai.services.project import _SUITE_LABEL_REGEX, Suite
import pytest

pytestmark = pytest.mark.small


def test_every_suite_is_either_mapped_or_deliberately_special():
    """No suite may be silently unmapped: it is either a label or a known exception."""
    special = {Suite.all, Suite.package}
    for suite in Suite:
        assert (suite in _SUITE_LABEL_REGEX) ^ (suite in special), (
            f"{suite.value!r} is neither a CTest label nor a handled special case; "
            "add it to _SUITE_LABEL_REGEX or branch on it in test()")


def test_package_is_not_a_ctest_label():
    """Indexing the mapping with `package` must not accidentally start working.

    If someone gives `package` a label regex, `test()` would still branch away
    to test_package() first and the label would be dead code that reads as live.
    """
    assert Suite.package not in _SUITE_LABEL_REGEX


def test_all_does_not_select_the_package_suite():
    """`all` runs every CTest test, and package is not one of them.

    Pinned because the alternative - quietly folding package into `all` - would
    make a full run several minutes slower (a full install plus a second
    configure and build) and, worse, make `all` mean two different things
    depending on whether the tree happened to be installable.
    """
    assert Suite.all not in _SUITE_LABEL_REGEX
    assert all(
        "package" not in regex for regex in _SUITE_LABEL_REGEX.values()
    ), "no CTest label group may claim to cover the package consumer"


def test_toolchain_args_exclude_the_prefix_path_and_the_source_siblings():
    """The shared helper must carry only what BOTH configures agree on.

    CMAKE_PREFIX_PATH differs legitimately (the consumer must also see the
    install prefix). SUSHIRUNTIME_DIR / SUSHIBLAS_DIR must never reach the
    consumer at all: handing it the sibling source trees is precisely the
    shortcut the package suite exists to forbid.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from sushiai.services.project import _toolchain_args

    # Every optional host tool is set, so the helper emits its largest flag list.
    cfg = SimpleNamespace(
        is_windows=True,
        ninja_exe="ninja.exe", pkgconf_exe="pkgconf.exe", rc_exe="rc.exe",
        vcpkg_triplet="x64-windows",
        expand=lambda value: value,
        resolved_compiler=lambda root: "clang++.exe",
        resolved_vcpkg=lambda root: "C:/vcpkg",
    )

    args = _toolchain_args(cfg, Path("."))
    assert args, "the helper must emit something to be worth sharing"
    for banned in ("CMAKE_PREFIX_PATH", "SUSHIRUNTIME_DIR", "SUSHIBLAS_DIR"):
        assert not any(banned in arg for arg in args), (
            f"{banned} must be set per-caller, not in the shared toolchain args")
