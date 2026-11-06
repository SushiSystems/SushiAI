# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# version: the version of the library as text and as numbers.

sushiai_add_module(NAME version TIER core
    SOURCES
        source/version.cpp)
