# aliases.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Holds the old spellings `sa` still answers to, each with its last day."""

from __future__ import annotations

from datetime import date

from sushicore.aliases import Alias, AliasTable

K_REMOVED_AFTER = date(2027, 3, 25)

K_ALIASES = AliasTable("sa", (
    Alias("sa doxygen", "sa docs", K_REMOVED_AFTER),
))
