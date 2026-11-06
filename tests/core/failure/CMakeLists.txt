# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module failure; see cmake/SushiAITestProgram.cmake.
sushiai_add_test_program(OWNER failure
    SOURCES
        test_check.cpp
        test_failure_block_counter.cpp
        test_in_memory_failure_block_counter.cpp
        test_in_memory_fatal_sink.cpp
        test_result.cpp
        test_status.cpp
    DEPENDS
        failure_expectations
        log)
