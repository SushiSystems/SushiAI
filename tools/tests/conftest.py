# conftest.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Registers the size marks of `testing` and sets each test's timeout from its size."""

from __future__ import annotations

import pytest

K_TIMEOUTS_IN_SECONDS = {"small": 1, "medium": 10, "large": 120}


def pytest_configure(config: pytest.Config) -> None:
    """Registers one mark for each size."""
    for size, seconds in K_TIMEOUTS_IN_SECONDS.items():
        config.addinivalue_line("markers", f"{size}: a {size} test, timed out after {seconds} s")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Gives each test the timeout of its size and fails collection of a test that carries none."""
    for item in items:
        class_size = getattr(getattr(item, "cls", None), "K_TEST_SIZE", None)
        module_size = getattr(getattr(item, "module", None), "K_TEST_SIZE", None)
        if class_size in K_TIMEOUTS_IN_SECONDS:
            item.add_marker(getattr(pytest.mark, class_size))
        elif not any(item.get_closest_marker(size) for size in K_TIMEOUTS_IN_SECONDS):
            if module_size in K_TIMEOUTS_IN_SECONDS:
                item.add_marker(getattr(pytest.mark, module_size))
        sizes = [size for size in K_TIMEOUTS_IN_SECONDS if item.get_closest_marker(size)]
        if len(sizes) != 1:
            raise pytest.UsageError(f"{item.nodeid} must carry exactly one size mark, found {sizes}")
        if item.config.pluginmanager.hasplugin("timeout"):
            item.add_marker(pytest.mark.timeout(K_TIMEOUTS_IN_SECONDS[sizes[0]]))
