"""CLI output for the SushiAI CLI.

Thin wrapper around :mod:`sushicore` — the actual theme/icon/renderer logic
(and its `[cli]` config schema) lives there and is shared with the
sushiruntime, sushiengine, sushiblas, and sushistack CLIs. See sushicore's
README to change colors.

The console is built lazily. Constructing it needs the repo's ``cli/`` config
directory, and locating that walks up for a ``CMakeLists.txt`` — which fails
outside a checkout. Doing that work at import time would make *every* ``sa``
invocation (including ``sa --help``, which Typer can serve without any
project) abort with "Not inside a SushiAI project". Deferring it via module
``__getattr__`` (PEP 562) means the console is materialised only when output
is actually produced, and even then it degrades to the default theme rather
than crashing when no config is reachable.
"""

from __future__ import annotations

from sushicore import build_console

from .config import config_dir

_console = None


def _get_console():
    """Build (once) and return the underlying sushicore console.

    Falls back to a config-less console when the ``cli/`` directory cannot be
    located (e.g. running outside a checkout), so output still works instead of
    raising SystemExit from config_dir().
    """
    global _console
    if _console is None:
        try:
            cfg_dir = config_dir()
            sources = [cfg_dir / "config.toml", cfg_dir / "config.local.toml"]
        except SystemExit:
            sources = []
        _console = build_console(sources)
    return _console


# Semantic helpers and the raw Rich console are resolved lazily on first access,
# so importing this module never touches the filesystem or the project layout.
_ATTRS = {
    "console",   # raw Rich console, for direct Table/Panel rendering
    "info", "success", "warn", "error",
    "command", "header", "fail_panel", "accent",
}


def __getattr__(name: str):
    if name in _ATTRS:
        c = _get_console()
        return c.console if name == "console" else getattr(c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
