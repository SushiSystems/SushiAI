# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module shape; see cmake/SushiAITestProgram.cmake.
sushiai_add_test_program(OWNER shape
    SOURCES
        test_shape.cpp
        test_tensor_shape.cpp
    DEPENDS
        failure_expectations)
