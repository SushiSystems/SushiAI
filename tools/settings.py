# settings.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Holds the settings of one repository that the shared checkers read.

copy_shared_tools.py writes this file only when a repository has none, and never overwrites it.
"""

from __future__ import annotations

K_TIER_ORDER: tuple[str, ...] = ("core", "layers", "vision", "language")
K_CLOSED = False
K_MACRO_PREFIXES: tuple[str, ...] = ("SA_", "SUSHIAI_")
K_DOMAIN_TIERS: tuple[str, ...] = ("vision", "language")
K_SAME_TIER_EDGES: frozenset[tuple[str, str]] = frozenset()
K_MODULE_FUNCTION = "sushiai_add_module"
K_PUBLIC_DEPENDS = "PUBLIC_DEPENDS"
K_PRIVATE_DEPENDS = "PRIVATE_DEPENDS"
K_INTERNAL_MARK = "INTERNAL"
K_PRIVATE_DEPENDS_ALSO: tuple[str, ...] = ()
K_LOG_PREFIX = "SA"
K_CONSOLE_PATHS: tuple[str, ...] = (
    "applications/",
    "cli/sushiai/services/",
    "modules/vision/darknet_forward_cases/",
    "modules/vision/darknet_training_cases/",
)
