# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module version; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER version
    SOURCES
        test_version.cpp)
