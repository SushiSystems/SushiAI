# aliases.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Holds the old spellings `sa` still answers to, each with its last day."""

from __future__ import annotations

from datetime import date

from sushicore.aliases import Alias, AliasTable

K_REMOVED_AFTER = date(2027, 3, 25)

K_ALIASES = AliasTable("sa", (
    Alias("sa doxygen", "sa docs", K_REMOVED_AFTER),
))
