# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module allocation; see cmake/SushiAITestProgram.cmake.
sushiai_add_test_program(OWNER allocation
    SOURCES
        test_allocation.cpp
        test_host_copy.cpp
    DEPENDS
        failure_expectations)
