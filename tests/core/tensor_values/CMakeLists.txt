# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module tensor_values; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER tensor_values
    SOURCES
        test_tensor_values.cpp
    DEPENDS
        tensor_values
        failure_expectations)
