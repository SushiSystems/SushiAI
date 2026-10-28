# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module umbrella; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/umbrella
    SOURCES
        test_vision/test_umbrella_reach.cpp
        test_vision/test_noexcept_surface.cpp)
