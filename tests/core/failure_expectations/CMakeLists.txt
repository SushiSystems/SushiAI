# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module failure_expectations; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER failure_expectations
    SOURCES
        test_status_expectations.cpp
    DEPENDS
        failure_expectations)
