# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module execution; see cmake/SushiAITestProgram.cmake.
sushiai_add_test_program(OWNER execution
    SOURCES
        compile_execution_backend.cpp
        compile_step_tasks.cpp
        test_execution.cpp
        test_host_execution_backend.cpp
        test_runtime_execution_backend.cpp
    DEPENDS
        allocation
        failure_expectations
        tensor_values)
