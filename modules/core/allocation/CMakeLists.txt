# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiai_add_module(NAME allocation TIER core
    SOURCES
        source/allocation.cpp
    PUBLIC_DEPENDS dtype failure shape
    PRIVATE_DEPENDS seam
    EXTERNAL_PUBLIC SushiBLAS::sushiblas SushiRuntime::core)
