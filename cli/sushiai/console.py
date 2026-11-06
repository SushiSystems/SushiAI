# console.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Exposes the SushiAI CLI's output helpers through a lazily built sushicore console.

The reasoning is in cli/README.md under "console.py".
"""

from __future__ import annotations

from sushicore.cli_console import LazyConsole

from .config import config_dir

_lazy = LazyConsole(config_dir)


def current():
    """Return the sushicore Console this run prints through, building it on first use."""
    return _lazy.get()


# PEP 562 resolves attributes on first access, so importing touches no filesystem.
__getattr__ = _lazy.attribute
