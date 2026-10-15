# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# engine_fixture: the Runtime a test program shares, its main() and EngineTest.

sushiai_add_module(NAME engine_fixture TIER core INTERNAL
    SOURCES
        source/test_main.cpp
    PUBLIC_DEPENDS core
    EXTERNAL_PUBLIC GTest::gtest)
