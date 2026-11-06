# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiai_add_module(NAME failure TIER core
    SOURCES
        source/check.cpp
        source/status.cpp
    PRIVATE_DEPENDS log
    EXTERNAL_PUBLIC SushiLog::text)
