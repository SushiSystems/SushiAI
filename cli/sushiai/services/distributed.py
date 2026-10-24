# distributed.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the cache variable `sa build --distributed` sets and the entry it passes to CMake."""

from __future__ import annotations

#: The cache variable that builds the distributed layer, forwarded to SR_ and SB_ as well.
K_DISTRIBUTED_VARIABLE = "SA_ENABLE_DISTRIBUTED"


def distributed_define(distributed: bool | None) -> list[str]:
    """Returns the VAR=VALUE entry that sets SA_ENABLE_DISTRIBUTED.

    Args:
        distributed: True builds the distributed layer, False stops building it, None leaves
            the cached value.
    """
    if distributed is None:
        return []
    return [f"{K_DISTRIBUTED_VARIABLE}={'ON' if distributed else 'OFF'}"]
