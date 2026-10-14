# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module core; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER core
    SOURCES
        test_ops.cpp
        test_result.cpp
        test_seam.cpp
        test_shape.cpp
        test_status.cpp
    DEPENDS
        failure_expectations
        tensor_values)
