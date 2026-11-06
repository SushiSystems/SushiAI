# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiai_add_module(NAME seam TIER core
    SOURCES
        source/seam.cpp
    PUBLIC_DEPENDS failure
    EXTERNAL_PUBLIC SushiBLAS::sushiblas SushiRuntime::core)
