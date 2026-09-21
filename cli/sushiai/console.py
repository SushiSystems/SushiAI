"""CLI output for the SushiAI CLI.

Thin wrapper around :mod:`sushicore` — the theme/icon/renderer logic and its
``[cli]`` config schema live there, shared with every other Sushi* CLI. See
sushicore's README to change colors.

The console is built lazily, by :class:`sushicore.cli_console.LazyConsole`:
constructing it needs this repo's ``cli/`` directory, and locating that fails
outside a checkout, so building at import time would make every ``sa``
invocation abort with "Not inside a SushiAI project" — including ``sa --help``,
which Typer can serve with no project at all.
"""

from __future__ import annotations

from sushicore.cli_console import LazyConsole

from .config import config_dir

_lazy = LazyConsole(config_dir)


def current():
    """Return the sushicore Console this run prints through, building it on first use."""
    return _lazy.get()


# PEP 562: attributes resolve on first access, so importing this module never
# touches the filesystem or the project layout.
__getattr__ = _lazy.attribute
