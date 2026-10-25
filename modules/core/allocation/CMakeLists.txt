# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiai_add_module(NAME allocation TIER core
    SOURCES
        source/allocation.cpp
    PUBLIC_DEPENDS dtype failure shape
    PRIVATE_DEPENDS seam
    EXTERNAL_PUBLIC SushiAI::blas SushiRuntime::core)
