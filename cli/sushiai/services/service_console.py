# service_console.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Declares the console surface the services print through."""

from __future__ import annotations

from typing import Protocol

from sushicore.proc import ConsoleLike


class ServiceConsole(ConsoleLike, Protocol):
    """The lines a service prints: a header, information, success, and the lines of ConsoleLike."""

    def header(self, text: str) -> None:
        """Print the title of a command."""
        ...

    def info(self, text: str) -> None:
        """Print an information line."""
        ...

    def success(self, text: str) -> None:
        """Print a success line."""
        ...
