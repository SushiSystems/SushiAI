# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module seam; see cmake/SushiAITestProgram.cmake.
sushiai_add_test_program(OWNER seam
    SOURCES
        test_seam.cpp
    DEPENDS
        failure_expectations)
