# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module layer_rows; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER layer_rows
    SOURCES
        test_layer_registry.cpp
        test_minimal_case.cpp
    DEPENDS
        layer_rows
        failure_expectations)
