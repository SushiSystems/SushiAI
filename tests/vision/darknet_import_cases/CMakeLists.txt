# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module darknet_import_cases; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/darknet_import_cases
    SOURCES
        test_darknet_import.cpp
    DEPENDS
        vision/darknet_import_cases
        failure_expectations)
