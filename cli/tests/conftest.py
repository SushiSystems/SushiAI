"""Test fixtures for the SushiAI CLI.

`sushicore` (the shared presentation/config layer) is an unpublished sibling
package that is not available in a clean CI environment, yet `sushiai`
imports it at module load. To let these tests run standalone we inject a small
stub for the handful of `sushicore` symbols the package touches. The stubs mirror
the real contract closely enough for the pure-logic tests below (console
laziness, project-root resolution) without pulling in Rich or the real config
plumbing.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _install_sushicore_stub() -> None:
    if "sushicore" in sys.modules:
        return

    # --- sushicore (top level): build_console -------------------------------
    sushicore = types.ModuleType("sushicore")

    class _FakeConsole:
        """Records that it was built and no-ops every semantic helper."""

        def __init__(self, sources):
            self.sources = list(sources)
            self.console = self  # raw-console slot; print() is a no-op below

        def print(self, *a, **k):  # Rich console API surface used by callers
            pass

        # Semantic helpers used across the CLI.
        def info(self, *a, **k): pass
        def success(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass
        def command(self, *a, **k): pass
        def header(self, *a, **k): pass
        def fail_panel(self, *a, **k): pass

        # accent is read as a style string, not called.
        accent = "bold"

    def build_console(sources):
        return _FakeConsole(sources)

    sushicore.build_console = build_console

    # --- sushicore.config_base ----------------------------------------------
    config_base = types.ModuleType("sushicore.config_base")

    @dataclass
    class ToolConfig:
        cxx: str = ""
        generator: str = ""
        vcpkg_root: str = ""
        vs_vcvars: str = ""
        ninja_exe: str = ""
        cmake_exe: str = ""
        ctest_exe: str = ""
        rc_exe: str = ""
        pkgconf_exe: str = ""
        doxygen_exe: str = ""
        vcpkg_triplet: str = ""

        def expand(self, value: str) -> str:
            return value

    def load_tool_config(cls, sources, plat, env_overrides):
        # Return defaults; tests that need populated config set fields directly.
        return cls()

    def write_tool_section(path, section, header_lines):
        return Path(path)

    config_base.ToolConfig = ToolConfig
    config_base.load_tool_config = load_tool_config
    config_base.write_tool_section = write_tool_section

    # --- sushicore.workspace -------------------------------------------------
    workspace = types.ModuleType("sushicore.workspace")

    def has_marker(name):
        return lambda p: (p / name).is_file()

    def resolve_env_path(env_var):
        import os
        v = os.environ.get(env_var)
        return Path(v) if v else None

    def walk_up(start, predicate):
        cur = Path(start).resolve()
        for candidate in (cur, *cur.parents):
            if predicate(candidate):
                return candidate
        return None

    workspace.has_marker = has_marker
    workspace.resolve_env_path = resolve_env_path
    workspace.walk_up = walk_up

    sys.modules["sushicore"] = sushicore
    sys.modules["sushicore.config_base"] = config_base
    sys.modules["sushicore.workspace"] = workspace


_install_sushicore_stub()
