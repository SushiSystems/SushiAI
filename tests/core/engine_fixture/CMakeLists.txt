# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module engine_fixture; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER engine_fixture
    SOURCES
        test_engine_test.cpp
        test_lazy_shared.cpp
        test_test_environment.cpp
        test_test_graph.cpp
    DEPENDS
        failure_expectations)
