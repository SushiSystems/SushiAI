# memory_limit.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Caps the memory this process and the programs it starts may commit.

The reasoning is in cli/README.md under "memory_limit.py".
"""

from __future__ import annotations

import ctypes
import os
import sys

K_ENVIRONMENT_KEY = "SA_MEMORY_LIMIT_GB"
K_DEFAULT_SHARE = 0.5
K_BYTES_PER_GB = 1024 ** 3

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400


class LimitError(Exception):
    """Reports a limit that could not be read or applied."""


def physical_memory_bytes() -> int:
    """Returns the physical memory of this machine, or 0 when it cannot be read."""
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
                ("total_page_file", ctypes.c_uint64), ("available_page_file", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                ("available_extended", ctypes.c_uint64),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return int(status.total_physical)

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        return 0


def resolve_limit_bytes(option_gb: float | None, environment: dict[str, str],
                        physical_bytes: int) -> int:
    """Returns the limit in bytes, or 0 for none.

    Args:
        option_gb: The value of --memory-limit, or None when the flag was not given.
        environment: The variables to read SA_MEMORY_LIMIT_GB from.
        physical_bytes: The physical memory of the machine; 0 when unknown.

    Raises:
        LimitError: The option or the variable is not a number of at least zero.
    """
    stated = option_gb
    if stated is None and environment.get(K_ENVIRONMENT_KEY, "") != "":
        text = environment[K_ENVIRONMENT_KEY]
        try:
            stated = float(text)
        except ValueError:
            raise LimitError(f"{K_ENVIRONMENT_KEY} must be a number of gigabytes; got '{text}'.")

    if stated is None:
        return int(physical_bytes * K_DEFAULT_SHARE)
    if stated < 0:
        raise LimitError(f"A memory limit must be at least 0 gigabytes; got {stated}.")
    return int(stated * K_BYTES_PER_GB)


def apply_limit(limit_bytes: int) -> bool:
    """Caps what this process and every process it starts may commit together.

    Args:
        limit_bytes: The ceiling; 0 applies none.

    Returns:
        Whether a limit is in force. False for a ceiling of 0, and on a platform other than
        Windows, where this module has no mechanism that counts committed memory.

    Raises:
        LimitError: The operating system refused the limit.
    """
    if limit_bytes <= 0 or sys.platform != "win32":
        return False
    _apply_job_limit(limit_bytes)
    return True


def _apply_job_limit(limit_bytes: int) -> None:
    """Puts this process into a job object whose committed memory is capped."""

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time", ctypes.c_int64), ("per_job_user_time", ctypes.c_int64),
            ("limit_flags", ctypes.c_uint32),
            ("minimum_working_set", ctypes.c_size_t), ("maximum_working_set", ctypes.c_size_t),
            ("active_process_limit", ctypes.c_uint32), ("affinity", ctypes.c_size_t),
            ("priority_class", ctypes.c_uint32), ("scheduling_class", ctypes.c_uint32),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", _BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
            ("process_memory_limit", ctypes.c_size_t), ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise LimitError(f"CreateJobObject failed with error {ctypes.get_last_error()}.")

    limits = _ExtendedLimits()
    limits.basic.limit_flags = (
        _JOB_OBJECT_LIMIT_JOB_MEMORY | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
    limits.job_memory_limit = limit_bytes
    if not kernel.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
            ctypes.sizeof(limits)):
        raise LimitError(f"SetInformationJobObject failed with error {ctypes.get_last_error()}.")
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise LimitError(f"AssignProcessToJobObject failed with error {ctypes.get_last_error()}.")
