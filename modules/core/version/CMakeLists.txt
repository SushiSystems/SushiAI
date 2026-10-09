# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# version: the version of the library as text and as numbers.

sushiai_add_module(NAME version TIER core
    SOURCES
        source/version.cpp
    TESTS
        tests/test_version.cpp)
