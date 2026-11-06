# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module darknet_import_cases; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/darknet_import_cases
    SOURCES
        test_darknet_import.cpp
    DEPENDS
        vision/darknet_import_cases
        failure_expectations)
