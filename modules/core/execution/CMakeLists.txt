# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiai_add_module(NAME execution TIER core
    SOURCES
        source/execution.cpp
    PUBLIC_DEPENDS failure seam
    EXTERNAL_PUBLIC SushiBLAS::sushiblas SushiRuntime::api)
