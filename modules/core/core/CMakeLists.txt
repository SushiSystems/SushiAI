# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# core: the status contract, shapes, element types and the seam to SushiBLAS.

sushiai_add_module(NAME core TIER core
    SOURCES
        source/allocation.cpp
        source/check.cpp
        source/seam.cpp
        source/status.cpp
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime SushiBLAS::sushiblas)
