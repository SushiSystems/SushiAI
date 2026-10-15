# sanitizer.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the sanitizer sets `sa build --sanitize` accepts and their SA_SANITIZE values."""

from __future__ import annotations

import enum

#: The CMake cache variable cmake/SushiAISanitizers.cmake reads.
K_SANITIZE_VARIABLE = "SA_SANITIZE"


class Sanitizer(str, enum.Enum):
    """Names the sanitizer sets `sa build --sanitize` accepts."""

    none = "none"
    address_undefined = "address-undefined"
    thread = "thread"


#: Maps each sanitizer set to the SA_SANITIZE value it configures.
K_SANITIZE_VALUE = {
    Sanitizer.none: "",
    Sanitizer.address_undefined: "address,undefined",
    Sanitizer.thread: "thread",
}


def sanitize_define(sanitizer: Sanitizer | None) -> list[str]:
    """Returns the VAR=VALUE entry that sets SA_SANITIZE; None leaves the cached value."""
    if sanitizer is None:
        return []
    return [f"{K_SANITIZE_VARIABLE}={K_SANITIZE_VALUE[sanitizer]}"]
