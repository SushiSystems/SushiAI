# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# engine_fixture: the Runtime a test program shares, its main() and EngineTest.

sushiai_add_module(NAME engine_fixture TIER core INTERNAL
    SOURCES
        source/test_main.cpp
    PUBLIC_DEPENDS execution failure
    EXTERNAL_PUBLIC GTest::gtest SushiRuntime::api)
