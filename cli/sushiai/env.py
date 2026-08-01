"""Build-environment snapshotting.

A parent process cannot ``call vcvars64.bat`` and inherit the result. Instead we
run it in a child shell, dump the resulting environment, and reuse that
dictionary for every cmake/ctest subprocess we spawn. The snapshot is cached on
disk keyed by the configuration that produced it; an unchanged config skips the
shell entirely.

Like SushiEngine's `se`, SushiAI selects no SYCL toolchain of its own: it
builds with the shared bundled clang++. So the only environment work here is
loading MSVC on Windows (clang++ -fsycl needs the MSVC headers/libs) and
putting the toolchain bin/lib on PATH so the compiler and its SYCL runtime
libraries resolve at build and run time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import console
from .config import Config, find_project_root

CACHE_NAME = ".sushi_env.json"


def _cache_key(cfg: Config) -> str:
    material = json.dumps(
        {
            "platform": cfg.platform,
            "vs_vcvars": cfg.vs_vcvars,
            "sushiruntime_dir": cfg.sushiruntime_dir,
            "sushiblas_dir": cfg.sushiblas_dir,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _merge_env(base: dict[str, str], overlay: dict[str, str]) -> dict[str, str]:
    """Merge overlay into base, prepending PATH into the base's case variant.

    PATH is collapsed into the *single* case variant the base already uses (e.g.
    Windows' "Path"). Writing the overlay's "PATH" as a new key while base keeps
    "Path" leaves two conflicting entries; Windows then honours only one for child
    processes, so toolchain dirs added to the other silently vanish from DLL/exe
    resolution.
    """
    merged = dict(base)
    path_key = next((k for k in merged if k.upper() == "PATH"), None)
    for k, v in overlay.items():
        if k.upper() == "PATH":
            target = path_key or k
            existing = merged.get(target, "")
            merged[target] = v + os.pathsep + existing if existing else v
            path_key = target
        else:
            merged[k] = v
    return merged


def _parse_windows_set(output: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key:
                env[key] = value
    return env


def _snapshot_windows(cfg: Config) -> dict[str, str] | None:
    vcvars = cfg.expand(cfg.vs_vcvars)
    if not (vcvars and Path(vcvars).is_file()):
        return None
    console.info("Loading Visual Studio environment (vcvars64)...")
    # Pass the whole command as a single string, NOT as ["cmd", "/c", script]:
    # with a list, subprocess re-quotes each element and mangles the inner quotes
    # around the (space-containing) vcvars path, so cmd.exe sees the quoted path as
    # one unknown token and returns non-zero — silently dropping the VS environment.
    script = f'call "{vcvars}" && set'
    result = subprocess.run("cmd /c " + script, capture_output=True, text=True)
    if result.returncode != 0:
        console.warn("vcvars64 returned non-zero; using current env.")
        return None
    return _parse_windows_set(result.stdout)


def _prepend_path(env: dict[str, str], var: str, dirs: list[str]) -> None:
    """Prepend *dirs* to the (case-insensitive) *var* path entry of *env*."""
    if not dirs:
        return
    key = next((k for k in env if k.upper() == var.upper()), var)
    existing = env.get(key, "")
    env[key] = os.pathsep.join(dirs + ([existing] if existing else []))


def _inject_runtime_toolchain(cfg: Config, env: dict[str, str]) -> None:
    """Add the shared bundled clang++ bin/lib to a build/run environment.

    Neither platform ships a system SYCL compiler: SushiAI builds with the
    intel/llvm clang++ provisioned into the shared SushiStack dependency tree
    (or the SushiRuntime sibling's own bundled copy, standalone), so that bin
    must be on PATH (and, on Linux, its lib on LD_LIBRARY_PATH) for the
    compiler and its SYCL runtime libraries to resolve.
    """
    try:
        root = find_project_root()
    except SystemExit:
        return
    bundle = cfg.deps_dir(root) / "toolchains" / "llvm-sycl"
    bin_dir = bundle / "bin"
    lib_dir = bundle / "lib"
    if bin_dir.is_dir():
        _prepend_path(env, "PATH", [str(bin_dir)])  # clang++ and the SYCL runtime DLLs
    if not cfg.is_windows and lib_dir.is_dir():
        _prepend_path(env, "LD_LIBRARY_PATH", [str(lib_dir)])

    # On Windows the runtime DLL pulls in vcpkg-installed dependencies (hwloc),
    # so the vcpkg installed bin must be on PATH for sushiruntime.dll to load.
    if cfg.is_windows:
        vcpkg = cfg.resolved_vcpkg(root)
        if vcpkg:
            vcpkg_bin = Path(vcpkg) / "installed" / cfg.vcpkg_triplet / "bin"
            if vcpkg_bin.is_dir():
                _prepend_path(env, "PATH", [str(vcpkg_bin)])


def load_build_env(cfg: Config, build_dir: Path) -> dict[str, str]:
    """Return the environment for build/test/run subprocesses.

    Falls back to the current ``os.environ`` when no toolchain scripts are
    configured (e.g. when clang++ and MSVC are already on PATH).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_file = build_dir / CACHE_NAME
    key = _cache_key(cfg)

    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                env = _merge_env(dict(os.environ), cached["env"])
                _inject_runtime_toolchain(cfg, env)
                return env
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # fall through and rebuild

    snapshot = _snapshot_windows(cfg) if cfg.is_windows else None
    if not snapshot:
        env = dict(os.environ)
        _inject_runtime_toolchain(cfg, env)
        return env

    try:
        cache_file.write_text(json.dumps({"key": key, "env": snapshot}))
    except OSError:
        pass

    env = _merge_env(dict(os.environ), snapshot)
    _inject_runtime_toolchain(cfg, env)
    return env
