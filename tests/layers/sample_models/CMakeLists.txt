# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module sample_models; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER sample_models
    SOURCES
        test_two_output_model.cpp
    DEPENDS
        sample_models
        failure_expectations)
