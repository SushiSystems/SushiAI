# test_suites.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the Suite enum and the two suites that are not CTest labels.

The reasoning is in cli/tests/README.md under "test_suites.py".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recorded_tools import offline_config
from sushiai.services.project import _SUITE_LABELS, Suite, _toolchain_args

K_SPECIAL = {Suite.all, Suite.package}


@pytest.mark.small
@pytest.mark.parametrize("suite", list(Suite), ids=lambda suite: suite.value)
def test_every_suite_is_either_mapped_or_deliberately_special(suite):
    """No suite may be silently unmapped: it is either a label or a known exception."""
    assert (suite in _SUITE_LABELS) ^ (suite in K_SPECIAL), (
        f"{suite.value!r} is neither a CTest label nor a handled special case; "
        "add it to _SUITE_LABELS or branch on it in test()")


@pytest.mark.small
@pytest.mark.parametrize("suite", sorted(_SUITE_LABELS, key=lambda suite: suite.value),
                         ids=lambda suite: suite.value)
def test_no_ctest_label_group_claims_to_cover_the_package_consumer(suite):
    """No label group of a suite names the package consumer, which is not a CTest test.

    Folding it into `all` would make a full run several minutes slower and let `all` mean
    two different things depending on whether the tree is installable.
    """
    assert all(label.value != "package" for label in _SUITE_LABELS[suite])


@pytest.mark.small
@pytest.mark.parametrize("banned", ["CMAKE_PREFIX_PATH", "SUSHIRUNTIME_DIR", "SUSHIBLAS_DIR"])
def test_toolchain_args_exclude_what_each_configure_sets_for_itself(banned):
    """The shared helper carries only what both configures agree on.

    CMAKE_PREFIX_PATH differs legitimately (the consumer must also see the install prefix).
    SUSHIRUNTIME_DIR and SUSHIBLAS_DIR must never reach the consumer at all: handing it the
    sibling source trees is the shortcut the package suite exists to forbid.
    """
    cfg = offline_config(
        "windows", ninja_exe="ninja.exe", pkgconf_exe="pkgconf.exe", rc_exe="rc.exe",
        vcpkg_triplet="x64-windows")

    args = _toolchain_args(cfg, Path("."))

    assert args
    assert not any(banned in arg for arg in args)
