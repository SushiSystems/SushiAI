# timing.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Holds the one shape every results reader of the time reader returns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Timing:
    """Holds the time one test took in one run.

    *size* is the size the results file gives, a JUnit property or an NUnit category. *suite* is set only by
    a reader whose file names the size of a test apart from its name, such as NUnit; the time reader then
    judges the test by that size and not by the pattern of its name.
    """

    name: str
    seconds: float
    size: str | None = None
    suite: str | None = None
