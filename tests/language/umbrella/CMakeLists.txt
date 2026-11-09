# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module umbrella; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/umbrella
    SOURCES
        compile_language.cpp
        test_language/compile_noexcept_surface.cpp
    DEPENDS
        failure_expectations)
