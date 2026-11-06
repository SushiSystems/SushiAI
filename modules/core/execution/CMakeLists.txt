# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiai_add_module(NAME execution TIER core
    SOURCES
        source/execution.cpp
    PUBLIC_DEPENDS failure seam
    EXTERNAL_PUBLIC SushiBLAS::sushiblas SushiRuntime::api)
