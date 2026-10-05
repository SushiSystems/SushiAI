# conftest.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Test fixtures for the SushiAI CLI.

``sushiai`` imports :mod:`sushicore` -- the shared core underneath every Sushi*
CLI -- at module load, so the suite cannot even be collected without it.

These tests used to run against a hand-written stub of that package. The stub
was a liability twice over: it had to grow every time sushicore did, and a test
could end up pinning the imitation rather than the contract (one of them
asserted on an attribute only the fake had). sushicore now ships inside the
SushiStack repository, so the real thing is reachable -- installed in the
environment, or sitting next door -- and that is what these tests use.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _ensure_sushicore() -> None:
    """Make :mod:`sushicore` importable, or fail with a message that says how.

    Already installed (CI does `pip install -e <stack>/sushicore`) wins. Failing
    that, it is found on disk: a workspace puts it at ``<workspace>/sushicore``,
    a developer's flat checkout layout at ``<siblings>/sushistack/sushicore``.
    """
    if importlib.util.find_spec("sushicore") is not None:
        return

    repo = Path(__file__).resolve().parents[2]
    for candidate in (repo.parent / "sushicore", repo.parent / "sushistack" / "sushicore"):
        if (candidate / "sushicore" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            importlib.invalidate_caches()
            if importlib.util.find_spec("sushicore") is not None:
                return

    raise RuntimeError(
        "sushicore is not importable, so the sushiai CLI cannot be imported. "
        "It ships in the SushiStack repository: install it with "
        "`pip install -e <sushistack>/sushicore`, or place a SushiStack "
        "checkout next to this one."
    )


_ensure_sushicore()
